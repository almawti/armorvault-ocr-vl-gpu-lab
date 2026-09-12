#!/usr/bin/env python3
"""Run PP-OCRv5 plus deterministic field mapping on synthetic documents."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from bidi.algorithm import get_display
from huggingface_hub import hf_hub_download
from rapidocr import RapidOCR

from generate_synthetic_ocr_benchmark import main as generate_main


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
DATE_RE = re.compile(r"(?<!\d)(\d{1,4})[\s./-]+(\d{1,2})[\s./-]+(\d{1,4})(?!\d)")
IDENTIFIER_RE = re.compile(r"(?<![A-Z0-9])([A-Z]?[A-Z0-9]{6,14})(?![A-Z0-9])", re.I)

FIELD_SCHEMA = {
    "documentNumber": {
        "aliases": [
            "document number", "identity number", "id number", "card number",
            "passport no", "license no", "no", "number", "رقم الهوية",
            "رقم الإقامة", "رقم الوثيقة", "رقم البطاقة", "الرقم",
        ]
    },
    "birthDate": {
        "aliases": ["date of birth", "birth date", "dob", "تاريخ الميلاد", "تاريخ الولادة"]
    },
    "issueDate": {
        "aliases": ["issue date", "date of issue", "doi", "issued on", "تاريخ الاصدار", "تاريخ الإصدار"]
    },
    "expiryDate": {
        "aliases": [
            "expiry date", "expiration date", "date of expiry", "valid until",
            "doe", "تاريخ الانتهاء", "تاريخ انتهاء الصلاحية",
        ]
    },
}

DOCUMENT_TYPES = {
    "saudi_national_id": ["الهوية الوطنية", "national id"],
    "residence_permit": ["بطاقة مقيم", "residence permit"],
    "passport": ["passport", "جواز سفر"],
    "driving_license": ["driving license", "رخصة قيادة"],
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.translate(ARABIC_DIGITS)).lower()
    text = re.sub(r"[\u064b-\u065f\u0670ـ]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", text).strip()


def logical_arabic(text: str) -> str:
    return get_display(text) if len(ARABIC_RE.findall(text)) >= 2 else text


def parse_date(text: str) -> str | None:
    match = DATE_RE.search(text.translate(ARABIC_DIGITS))
    if not match:
        return None
    a, b, c = map(int, match.groups())
    candidates = [(c, b, a), (a, b, c)] if a <= 31 else [(a, b, c)]
    for year, month, day in candidates:
        if year < 100:
            year += 2000 if year < 50 else 1900
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def center(box) -> list[float]:
    points = np.asarray(box, dtype=float)
    return [float(points[:, 0].mean()), float(points[:, 1].mean())]


def distance(left: dict, right: dict) -> float:
    dx = abs(right["center"][0] - left["center"][0])
    dy = abs(right["center"][1] - left["center"][1])
    return math.hypot(dx, dy * 2.2) * (0.45 if dy <= 30 else 1.0)


def has_alias(line: dict, aliases: list[str]) -> bool:
    return any(normalize(alias) in line["normalized"] for alias in aliases)


def identifier(text: str) -> str | None:
    cleaned = text.translate(ARABIC_DIGITS).upper()
    candidates = [value for value in IDENTIFIER_RE.findall(cleaned) if not value.isalpha()]
    return max(candidates, key=len, default=None)


def extract(lines: list[dict]) -> tuple[dict, dict]:
    result = {"issueDate": None}
    evidence = {}
    all_text = " ".join(line["normalized"] for line in lines)
    result["documentType"] = next(
        (kind for kind, aliases in DOCUMENT_TYPES.items() if any(normalize(alias) in all_text for alias in aliases)),
        "other",
    )

    for field, spec in FIELD_SCHEMA.items():
        labels = [line for line in lines if has_alias(line, spec["aliases"])]
        values = []
        for line in lines:
            value = parse_date(line["text"]) if field != "documentNumber" else identifier(line["text"])
            if value:
                values.append((line, value))
        ranked = []
        for label in labels:
            for value_line, value in values:
                inline = label["id"] == value_line["id"]
                ranked.append((not inline, 0.0 if inline else distance(label, value_line), label, value_line, value))
        if ranked:
            _, _, label, value_line, value = min(ranked, key=lambda item: item[:2])
            result[field] = value
            evidence[field] = [label["id"]] if label["id"] == value_line["id"] else [label["id"], value_line["id"]]
        elif field not in result:
            result[field] = None

    latin_names = [
        line for line in lines
        if re.fullmatch(r"[A-Z][A-Z, .-]{8,}", line["text"])
        and len(line["text"].replace(",", " ").split()) >= 3
        and not has_alias(line, ["full name", "name"])
    ]
    latin = max(latin_names, key=lambda line: line["confidence"], default=None)
    result["holderNameEnglish"] = latin["text"] if latin else None

    excluded = [normalize(value) for value in (
        "الهوية الوطنية", "المملكة العربية السعودية", "وزارة الداخلية",
        "اسم حامل الوثيقة", "الاسم الكامل", "تاريخ الميلاد",
        "تاريخ الانتهاء", "مكان الميلاد", "رقم الهوية", "بطاقة مقيم",
    )]
    arabic_names = []
    for line in lines:
        words = line["normalized"].split()
        if len(words) < 3 or sum(bool(ARABIC_RE.search(word)) for word in words) < 3:
            continue
        if any(value in line["normalized"] for value in excluded):
            continue
        score = line["confidence"]
        if normalize("بنت") in words:
            score += 3
        elif normalize("بن") in words:
            score += 2
        if latin:
            score += max(0, 1.5 - abs(line["center"][1] - latin["center"][1]) / 80)
        arabic_names.append((score, line))
    arabic = max(arabic_names, key=lambda item: item[0], default=(0, None))[1]
    result["holderNameArabic"] = arabic["text"] if arabic else None
    return result, evidence


def make_ocr(workdir: Path) -> RapidOCR:
    detector = hf_hub_download("PaddlePaddle/PP-OCRv5_mobile_det_onnx", "inference.onnx")
    recognizer = hf_hub_download("PaddlePaddle/arabic_PP-OCRv5_mobile_rec_onnx", "inference.onnx")
    config = hf_hub_download("PaddlePaddle/arabic_PP-OCRv5_mobile_rec_onnx", "inference.yml")
    characters = yaml.safe_load(Path(config).read_text(encoding="utf-8"))["PostProcess"]["character_dict"]
    keys = workdir / "ppocrv5_arabic_dict.txt"
    keys.write_text("\n".join(map(str, characters)), encoding="utf-8")
    return RapidOCR(params={
        "Det.model_path": detector,
        "Rec.model_path": recognizer,
        "Rec.rec_keys_path": str(keys),
        "Rec.rec_img_shape": [3, 48, 320],
    })


def run_ocr(ocr: RapidOCR, image: Path) -> tuple[list[dict], float]:
    started = time.perf_counter()
    output = ocr(str(image), use_cls=False)
    elapsed = time.perf_counter() - started
    texts = list(output.txts or [])
    scores = [float(score) for score in (output.scores or [])]
    boxes = output.boxes.tolist() if output.boxes is not None else []
    lines = []
    for index, text in enumerate(texts):
        corrected = logical_arabic(text)
        lines.append({
            "id": index,
            "rawText": text,
            "text": corrected,
            "normalized": normalize(corrected),
            "confidence": round(scores[index], 4),
            "center": center(boxes[index]),
        })
    return lines, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/content/armorvault-synthetic-benchmark")
    parser.add_argument("--output", default="/content/armorvault-synthetic-results.json")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    if not (dataset / "manifest.json").exists():
        import sys
        old_argv = sys.argv
        sys.argv = ["generate", "--output", str(dataset)]
        try:
            generate_main()
        finally:
            sys.argv = old_argv

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    ocr = make_ocr(dataset)
    reports = []
    exact = total = 0
    started = time.perf_counter()
    for item in manifest["documents"]:
        lines, seconds = run_ocr(ocr, dataset / item["file"])
        actual, evidence = extract(lines)
        comparison = {}
        for field, expected in item["expected"].items():
            matched = actual.get(field) == expected
            exact += int(matched)
            total += 1
            comparison[field] = {"actual": actual.get(field), "expected": expected, "exactMatch": matched}
        reports.append({
            "file": item["file"], "layout": item["layout"], "variant": item["variant"],
            "ocrSeconds": round(seconds, 3), "comparison": comparison,
            "evidence": evidence, "ocrLines": lines,
        })
        print(f"{item['file']}: {sum(v['exactMatch'] for v in comparison.values())}/{len(comparison)}")

    result = {
        "models": {"ocr": "PP-OCRv5 Mobile detector + Arabic recognizer ONNX", "semanticMatcher": None},
        "documents": len(reports), "exactFields": exact, "totalFields": total,
        "fieldAccuracy": round(exact / total, 4) if total else 0,
        "totalSeconds": round(time.perf_counter() - started, 3), "reports": reports,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFINAL BENCHMARK RESULT")
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()