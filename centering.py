#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
import cv2
import numpy as np

# --------- ЖЁСТКИЕ ПУТИ (поменяйте при необходимости) ----------
IMG_DIR  = Path(r'D:\Avocation\Sky\80. 07-08.09.2025\jpg')       # исходные изображения
TXT_DIR  = Path(r'D:\Avocation\Sky\80. 07-08.09.2025\detected\coords')      # метаданные: cx, cy
OUT_DIR  = Path(r'D:\Avocation\Sky\80. 07-08.09.2025/padded_jpg')    # куда писать результат
# ---------------------------------------------------------------

# Какие расширения считать картинками
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

cx_re = re.compile(r"cx\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
cy_re = re.compile(r"cy\s*=\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

def parse_cx_cy(txt_path: Path):
    """
    Парсит файл с вида:
      cx=3023.500000, cy=1341.500000, R=..., method=...
    Возвращает (cx, cy) в пикселях (float).
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    m_cx = cx_re.search(text)
    m_cy = cy_re.search(text)
    if not (m_cx and m_cy):
        raise ValueError(f"Не нашёл cx/cy в {txt_path}")
    cx = float(m_cx.group(1))
    cy = float(m_cy.group(1))
    return cx, cy

def compute_one_sided_padding(size: int, c: float):
    """
    Для одной оси вычисляет паддинг ТОЛЬКО с одной стороны.
    size — исходный размер по оси (W или H)
    c    — координата центра по оси (cx или cy) в пикселях (отсчёт от 0 слева/сверху)

    Если c < size/2 -> добавляем паддинг слева/сверху
    Если c > size/2 -> добавляем паддинг справа/снизу
    Если ровно по центру -> паддинг 0
    Возвращает пару целых (pad_before, pad_after)
    """
    half = size / 2.0
    if c < half:
        pad_before = int(round(size - 2.0 * c))  # слева/сверху
        pad_before = max(pad_before, 0)
        pad_after = 0
    elif c > half:
        pad_after = int(round(2.0 * c - size))   # справа/снизу
        pad_after = max(pad_after, 0)
        pad_before = 0
    else:
        pad_before = pad_after = 0
    return pad_before, pad_after

def pad_to_center(img: np.ndarray, cx: float, cy: float):
    """
    Возвращает новое изображение с паддингами только с одной стороны по каждой оси,
    чтобы точка (cx, cy) стала центром.
    Паддинги — репликация крайних пикселей (cv2.BORDER_REPLICATE).
    """
    h, w = img.shape[:2]

    # Горизонталь (x): left/right
    pad_left, pad_right = compute_one_sided_padding(w, cx)
    # Вертикаль (y): top/bottom
    pad_top, pad_bottom = compute_one_sided_padding(h, cy)

    # Выполняем паддинг
    padded = cv2.copyMakeBorder(
        img,
        top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=cv2.BORDER_REPLICATE
    )

    # Проверка (необязательно): насколько центр действительно в центре
    new_h, new_w = padded.shape[:2]
    new_cx = cx + pad_left
    new_cy = cy + pad_top
    # Если очень хочется убедиться, можно раскомментировать:
    # assert abs(new_cx - new_w / 2.0) <= 0.5, (new_cx, new_w/2.0)
    # assert abs(new_cy - new_h / 2.0) <= 0.5, (new_cy, new_h/2.0)

    return padded, (pad_left, pad_right, pad_top, pad_bottom)

def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"Нет папки с изображениями: {IMG_DIR}")
    if not TXT_DIR.exists():
        raise FileNotFoundError(f"Нет папки с метаданными: {TXT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = [p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    images.sort()
    total = 0
    ok = 0
    skipped = 0

    for img_path in images:
        total += 1
        stem = img_path.stem
        txt_path = TXT_DIR / f"{stem}.txt"
        if not txt_path.exists():
            print(f"[SKIP] Нет txt для {img_path.name}")
            skipped += 1
            continue

        try:
            cx, cy = parse_cx_cy(txt_path)
        except Exception as e:
            print(f"[SKIP] Ошибка парсинга {txt_path.name}: {e}")
            skipped += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[SKIP] Не удалось прочитать изображение: {img_path.name}")
            skipped += 1
            continue

        try:
            padded, pads = pad_to_center(img, cx, cy)
        except Exception as e:
            print(f"[SKIP] Ошибка паддинга {img_path.name}: {e}")
            skipped += 1
            continue

        out_path = OUT_DIR / img_path.name
        success = cv2.imwrite(str(out_path), padded)
        if not success:
            print(f"[ERR ] Не удалось сохранить: {out_path.name}")
            skipped += 1
            continue

        ok += 1
        pl, pr, pt, pb = pads
        print(f"[OK  ] {img_path.name} -> {out_path.name} | pads L{pl} R{pr} T{pt} B{pb}")

    print("\nГотово.")
    print(f"Всего: {total}, успешно: {ok}, пропущено: {skipped}")
    print(f"Результаты: {OUT_DIR}")

if __name__ == "__main__":
    main()
