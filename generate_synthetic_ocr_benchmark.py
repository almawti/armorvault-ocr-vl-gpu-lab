#!/usr/bin/env python3
"""Generate privacy-safe synthetic documents and field-level ground truth."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None
    get_display = None


WIDTH, HEIGHT = 1200, 760
BG = (241, 244, 239)
INK = (23, 35, 44)
ACCENT = (30, 91, 79)

CASES = [
    {
        "id": "saudi-id-ar-en",
        "type": "saudi_national_id",
        "title": "الهوية الوطنية | NATIONAL ID",
        "fields": [
            ("الاسم", "نورة بنت عبدالرحمن أحمد", "holderNameArabic"),
            ("NAME", "NOURAH ABDULRAHMAN AHMAD", "holderNameEnglish"),
            ("رقم الهوية", "1234567890", "documentNumber"),
            ("تاريخ الميلاد", "28/04/1992", "birthDate"),
            ("تاريخ الانتهاء", "19/01/2032", "expiryDate"),
        ],
    },
    {
        "id": "residence-card-ar",
        "type": "residence_permit",
        "title": "بطاقة مقيم",
        "fields": [
            ("الاسم الكامل", "سارة بنت محمد علي", "holderNameArabic"),
            ("رقم الإقامة", "2457819630", "documentNumber"),
            ("تاريخ الميلاد", "07/11/1996", "birthDate"),
            ("تاريخ الإصدار", "15/03/2024", "issueDate"),
            ("تاريخ الانتهاء", "15/03/2026", "expiryDate"),
        ],
    },
    {
        "id": "passport-en",
        "type": "passport",
        "title": "PASSPORT",
        "fields": [
            ("FULL NAME", "OMAR ABDULLAH SALEH", "holderNameEnglish"),
            ("PASSPORT NO", "P8472951", "documentNumber"),
            ("DATE OF BIRTH", "14/08/1988", "birthDate"),
            ("DATE OF ISSUE", "09/05/2023", "issueDate"),
            ("DATE OF EXPIRY", "09/05/2033", "expiryDate"),
        ],
    },
    {
        "id": "driving-license-mixed",
        "type": "driving_license",
        "title": "رخصة قيادة | DRIVING LICENSE",
        "fields": [
            ("الاسم", "خالد بن يوسف حسن", "holderNameArabic"),
            ("NAME", "KHALID YOUSIF HASSAN", "holderNameEnglish"),
            ("LICENSE NO", "D55190327", "documentNumber"),
            ("DOB", "03/12/1990", "birthDate"),
            ("VALID UNTIL", "22/06/2029", "expiryDate"),
        ],
    },
]


def fonts(font_path: str) -> tuple[ImageFont.FreeTypeFont, ...]:
    return tuple(ImageFont.truetype(font_path, size) for size in (46, 31, 34))


def draw_rtl(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=INK):
    try:
        draw.text(xy, text, font=font, fill=fill, anchor="ra", direction="rtl", language="ar")
    except (KeyError, TypeError, ValueError):
        rendered = get_display(arabic_reshaper.reshape(text)) if arabic_reshaper and get_display else text
        draw.text(xy, rendered, font=font, fill=fill, anchor="ra")


def render(case: dict, layout: str, font_path: str) -> Image.Image:
    title_font, label_font, value_font = fonts(font_path)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((25, 25, WIDTH - 25, HEIGHT - 25), 36, outline=ACCENT, width=5)
    draw.text((60, 55), case["title"], font=title_font, fill=ACCENT)
    draw.line((60, 125, WIDTH - 60, 125), fill=ACCENT, width=3)

    for index, (label, value, _) in enumerate(case["fields"]):
        y = 160 + index * 105
        is_arabic = any("\u0600" <= char <= "\u06ff" for char in label + value)
        if layout == "inline":
            text = f"{label}: {value}"
            if is_arabic:
                draw_rtl(draw, (WIDTH - 75, y), text, value_font)
            else:
                draw.text((75, y), text, font=value_font, fill=INK)
        elif layout == "stacked":
            x = WIDTH - 75 if is_arabic else 75
            if is_arabic:
                draw_rtl(draw, (x, y), label, label_font, ACCENT)
                draw_rtl(draw, (x, y + 38), value, value_font)
            else:
                draw.text((x, y), label, font=label_font, fill=ACCENT)
                draw.text((x, y + 38), value, font=value_font, fill=INK)
        else:
            if is_arabic:
                draw_rtl(draw, (WIDTH - 75, y), label, label_font, ACCENT)
                draw_rtl(draw, (650, y), value, value_font)
            else:
                draw.text((75, y), label, font=label_font, fill=ACCENT)
                draw.text((475, y), value, font=value_font, fill=INK)
    return image


def degrade(image: Image.Image, variant: str) -> Image.Image:
    if variant == "clear":
        return image
    if variant == "tilted":
        return image.rotate(2.2, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(220, 220, 215))
    result = ImageEnhance.Contrast(image).enhance(0.58)
    result = ImageEnhance.Brightness(result).enhance(0.82)
    return result.filter(ImageFilter.GaussianBlur(0.7))


def expected(case: dict) -> dict:
    values = {"documentType": case["type"], "issueDate": None}
    for _, value, field in case["fields"]:
        if field in {"birthDate", "issueDate", "expiryDate"}:
            values[field] = datetime.strptime(value, "%d/%m/%Y").strftime("%Y-%m-%d")
        else:
            values[field] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="synthetic-benchmark")
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    args = parser.parse_args()

    random.seed(42)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    layouts = ("inline", "stacked", "columns")
    variants = ("clear", "tilted", "low-contrast")

    for index, case in enumerate(CASES):
        layout = layouts[index % len(layouts)]
        for variant in variants:
            filename = f"{case['id']}__{layout}__{variant}.png"
            degrade(render(case, layout, args.font), variant).save(output / filename)
            manifest.append(
                {
                    "file": filename,
                    "caseId": case["id"],
                    "layout": layout,
                    "variant": variant,
                    "expected": expected(case),
                }
            )

    (output / "manifest.json").write_text(
        json.dumps({"version": 1, "documents": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(manifest)} documents in {output}")


if __name__ == "__main__":
    main()