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

NAME_ALIASES = {
    "holderNameArabic": ["اسم حامل الوثيقة", "الاسم الكامل", "الاسم"],
    "holderNameEnglish": ["full name", "name in english", "holder name", "name"],
}
ALL_FIELD_ALIASES = [
    alias
    for spec in list(FIELD_SCHEMA.values()) + [{"aliases": aliases} for aliases in NAME_ALIASES.values()]
    for alias in spec["aliases"]
]


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
    normalized = line["normalized"]
    compact = normalized.replace(" ", "")
    return any(
        normalize(alias) in normalized or normalize(alias).replace(" ", "") in compact
        for alias in aliases
    )


def strip_leading_alias(text: str, aliases: list[str]) -> str | None:
    """Return an inline value after a label, including joined Arabic labels."""
    cleaned = text.strip(" :：-|")
    normalized_cleaned = normalize(cleaned)
    if any(normalized_cleaned == normalize(alias) for alias in aliases):
        return None
    for alias in sorted(aliases, key=len, reverse=True):
        pattern = re.compile(rf"^\s*{re.escape(alias)}\s*[:：|\-]?\s*", re.I)
        value = pattern.sub("", cleaned, count=1).strip(" :：-|")
        if value != cleaned and value:
            return value
    normalized_compact = normalize(cleaned).replace(" ", "")
    for alias in sorted(aliases, key=len, reverse=True):
        compact_alias = normalize(alias).replace(" ", "")
        if normalized_compact.startswith(compact_alias) and len(normalized_compact) > len(compact_alias):
            target = normalize(alias).split()[-1]
            match = re.search(re.escape(target), normalize(cleaned))
            if match:
                normalized_value = normalize(cleaned)[match.end():].strip()
                if normalized_value:
                    return normalized_value
    return None


def identifier(text: str) -> str | None:
    cleaned = text.translate(ARABIC_DIGITS).upper()
    candidates = [value for value in IDENTIFIER_RE.findall(cleaned) if not value.isalpha()]
    return max(candidates, key=len, default=None)


def extract_name(lines: list[dict], field: str) -> tuple[str | None, list[int]]:
    aliases = NAME_ALIASES[field]
    labels = [line for line in lines if has_alias(line, aliases)]
    is_arabic = field == "holderNameArabic"

    for label in labels:
        value = strip_leading_alias(label["text"], aliases)
        if not value:
            continue
        if is_arabic and len(ARABIC_RE.findall(value)) >= 3:
            return value, [label["id"]]
        if not is_arabic and re.fullmatch(r"[A-Z][A-Z, .-]{5,}", value.upper()):
            return value.upper(), [label["id"]]

    candidates = []
    for line in lines:
        if has_alias(line, ALL_FIELD_ALIASES):
            continue
        words = line["text"].replace(",", " ").split()
        if is_arabic:
            valid = len(words) >= 3 and sum(bool(ARABIC_RE.search(word)) for word in words) >= 3
        else:
            valid = (
                len(words) >= 3
                and re.fullmatch(r"[A-Z][A-Z, .-]{5,}", line["text"].upper()) is not None
                and not ARABIC_RE.search(line["text"])
            )
        if valid:
            candidates.append(line)
    ranked = [
        (distance(label, candidate), -candidate["confidence"], label, candidate)
        for label in labels for candidate in candidates
    ]
    if ranked:
        _, _, label, candidate = min(ranked, key=lambda item: item[:2])
        return candidate["text"], [label["id"], candidate["id"]]
    if candidates:
        candidate = max(candidates, key=lambda line: line["confidence"])
        return candidate["text"], [candidate["id"]]
    return None, []


def extract(lines: list[dict]) -> tuple[dict, dict, list[dict], dict]:
    result = {"issueDate": None}
    evidence = {}
    review = []
    candidate_diagnostics = {}
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
        candidate_diagnostics[field] = {
            "labels": [
                {
                    "id": line["id"],
                    "text": line["text"],
                    "confidence": line["confidence"],
                    "center": line["center"],
                }
                for line in labels
            ],
            "compatibleValues": [
                {
                    "id": line["id"],
                    "value": value,
                    "text": line["text"],
                    "confidence": line["confidence"],
                    "center": line["center"],
                }
                for line, value in values
            ],
        }
        ranked = []
        for label in labels:
            for value_line, value in values:
                inline = label["id"] == value_line["id"]
                ranked.append((not inline, 0.0 if inline else distance(label, value_line), label, value_line, value))
        if ranked:
            _, _, label, value_line, value = min(ranked, key=lambda item: item[:2])
            result[field] = value
            evidence[field] = [label["id"]] if label["id"] == value_line["id"] else [label["id"], value_line["id"]]
            if field != "documentNumber" and value_line["confidence"] < 0.90:
                review.append({
                    "field": field,
                    "value": value,
                    "reason": "low OCR confidence",
                    "confidence": value_line["confidence"],
                    "sourceLineId": value_line["id"],
                })
        elif field not in result:
            result[field] = None

    for field in ("holderNameEnglish", "holderNameArabic"):
        value, source_ids = extract_name(lines, field)
        result[field] = value
        if source_ids:
            evidence[field] = source_ids
        candidate_diagnostics[field] = {
            "labels": [
                {
                    "id": line["id"],
                    "text": line["text"],
                    "confidence": line["confidence"],
                    "center": line["center"],
                }
                for line in lines if has_alias(line, NAME_ALIASES[field])
            ],
            "selectedEvidence": source_ids,
        }
    if result.get("expiryDate"):
        expiry = datetime.fromisoformat(result["expiryDate"]).date()
        if expiry < datetime.now().date():
            review.append({
                "field": "expiryDate",
                "value": result["expiryDate"],
                "reason": "expiry date is in the past",
            })
    date_values = [
        (field, result.get(field))
        for field in ("birthDate", "issueDate", "expiryDate")
        if result.get(field)
    ]
    for index, (left_field, left_value) in enumerate(date_values):
        for right_field, right_value in date_values[index + 1:]:
            if left_value >= right_value:
                review.append({
                    "field": right_field,
                    "value": right_value,
                    "reason": f"date chronology conflicts with {left_field}",
                })
    return result, evidence, review, candidate_diagnostics


def expected_visible_in_ocr(field: str, expected, lines: list[dict]) -> bool:
    if expected is None:
        return True
    if field == "documentType":
        aliases = DOCUMENT_TYPES.get(str(expected), [])
        return any(has_alias(line, aliases) for line in lines)
    if field in {"birthDate", "issueDate", "expiryDate"}:
        return any(parse_date(line["text"]) == expected for line in lines)
    if field == "documentNumber":
        return any(identifier(line["text"]) == expected for line in lines)
    expected_compact = normalize(str(expected)).replace(" ", "")
    return any(expected_compact in line["normalized"].replace(" ", "") for line in lines)


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
    field_stats = {}
    failure_breakdown = {"ocr": 0, "mapping": 0}
    started = time.perf_counter()
    for item in manifest["documents"]:
        lines, seconds = run_ocr(ocr, dataset / item["file"])
        actual, evidence, review, candidate_diagnostics = extract(lines)
        comparison = {}
        for field, expected in item["expected"].items():
            matched = actual.get(field) == expected
            exact += int(matched)
            total += 1
            stats = field_stats.setdefault(field, {"correct": 0, "total": 0})
            stats["correct"] += int(matched)
            stats["total"] += 1
            failure_type = None
            if not matched:
                failure_type = "mapping" if expected_visible_in_ocr(field, expected, lines) else "ocr"
                failure_breakdown[failure_type] += 1
            comparison[field] = {
                "actual": actual.get(field),
                "expected": expected,
                "exactMatch": matched,
                "failureType": failure_type,
            }
        reports.append({
            "file": item["file"], "layout": item["layout"], "variant": item["variant"],
            "ocrSeconds": round(seconds, 3), "comparison": comparison,
            "evidence": evidence,
            "candidateDiagnostics": candidate_diagnostics,
            "reviewRequired": review,
            "ocrLines": lines,
        })
        print(f"{item['file']}: {sum(v['exactMatch'] for v in comparison.values())}/{len(comparison)}")

    for stats in field_stats.values():
        stats["accuracy"] = round(stats["correct"] / stats["total"], 4) if stats["total"] else 0
    result = {
        "models": {"ocr": "PP-OCRv5 Mobile detector + Arabic recognizer ONNX", "semanticMatcher": None},
        "documents": len(reports), "exactFields": exact, "totalFields": total,
        "fieldAccuracy": round(exact / total, 4) if total else 0,
        "accuracyByField": field_stats,
        "failureBreakdown": failure_breakdown,
        "reviewRequiredCount": sum(len(report["reviewRequired"]) for report in reports),
        "totalSeconds": round(time.perf_counter() - started, 3), "reports": reports,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFINAL BENCHMARK RESULT")
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, ensure_ascii=False, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()