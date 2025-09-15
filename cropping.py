#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from pathlib import Path

# --------- ЖЁСТКИЕ ПУТИ ----------
PADDED_DIR = Path(r'D:\Avocation\Sky\80. 07-08.09.2025/padded_jpg')      # вход: изображения с паддингами
OUT_DIR    = Path(r'D:\Avocation\Sky\80. 07-08.09.2025\cropped_1080')      # выход: центр 1080x1080
# ---------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CROP_SIZE = 1080
HALF = CROP_SIZE // 2

def ensure_min_size(img, min_side=1080):
    """
    Если по любой оси размер меньше min_side — добавляем симметричный
    реплицированный бордер, чтобы обе стороны были >= min_side.
    Это не меняет масштаб и сохраняет центр.
    """
    h, w = img.shape[:2]
    pad_h = max(0, min_side - h)
    pad_w = max(0, min_side - w)

    if pad_h == 0 and pad_w == 0:
        return img, (0,0,0,0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_REPLICATE
    )
    return padded, (left, right, top, bottom)

def center_crop_1080(img):
    """
    Центрированный кроп 1080x1080 вокруг геометрического центра кадра.
    Предполагается, что по обеим осям размер >= 1080 (если нет — предварительно допаддим).
    """
    h, w = img.shape[:2]
    cx = w // 2
    cy = h // 2

    x0 = cx - HALF
    y0 = cy - HALF
    x1 = x0 + CROP_SIZE
    y1 = y0 + CROP_SIZE

    # На всякий случай подправим границы (после ensure_min_size не должно понадобиться)
    if x0 < 0: 
        x0, x1 = 0, CROP_SIZE
    if y0 < 0: 
        y0, y1 = 0, CROP_SIZE
    if x1 > w: 
        x1, x0 = w, w - CROP_SIZE
    if y1 > h: 
        y1, y0 = h, h - CROP_SIZE

    crop = img[y0:y1, x0:x1]
    # Гарантируем точный размер
    if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
        # Последняя страховка: если вдруг из-за нечётных размеров/индексов,
        # чуть сдвинем окно и/или доклеим репликацию.
        need_h = CROP_SIZE - crop.shape[0]
        need_w = CROP_SIZE - crop.shape[1]
        top = max(0, need_h // 2)
        bottom = max(0, need_h - top)
        left = max(0, need_w // 2)
        right = max(0, need_w - left)
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REPLICATE)
        crop = crop[:CROP_SIZE, :CROP_SIZE]

    return crop

def main():
    if not PADDED_DIR.exists():
        raise FileNotFoundError(f"Нет папки с входными изображениями: {PADDED_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in PADDED_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    total, ok, skipped = 0, 0, 0
    for img_path in images:
        total += 1
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[SKIP] Не удалось прочитать: {img_path.name}")
            skipped += 1
            continue

        # Дотягиваем до минимального размера при необходимости
        img2, pads = ensure_min_size(img, min_side=CROP_SIZE)

        try:
            crop = center_crop_1080(img2)
        except Exception as e:
            print(f"[SKIP] Ошибка кропа {img_path.name}: {e}")
            skipped += 1
            continue

        out_path = OUT_DIR / img_path.name  # сохраняем то же имя/расширение
        if not cv2.imwrite(str(out_path), crop):
            print(f"[ERR ] Не удалось сохранить: {out_path.name}")
            skipped += 1
            continue

        ok += 1
        l, r, t, b = pads
        pad_info = f" | pad L{l} R{r} T{t} B{b}" if (l or r or t or b) else ""
        print(f"[OK  ] {img_path.name} -> {out_path.name}{pad_info}")

    print("\nГотово.")
    print(f"Всего: {total}, успешно: {ok}, пропущено: {skipped}")
    print(f"Результаты: {OUT_DIR}")

if __name__ == "__main__":
    main()
