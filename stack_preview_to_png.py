#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
from tifffile import imread
from PIL import Image
from astropy.io import fits
from tqdm import tqdm

# ===== ЖЁСТКИЕ ПУТИ/ПАРАМЕТРЫ =====
STACK_DIR  = r'D:\Avocation\Sky\80. 07-08.09.2025\stack\stack_out'      # где лежат конечные FITS/TIFF после стека
OUTPUT_DIR = r'D:\Avocation\Sky\80. 07-08.09.2025\stack\preview_png'

# Что обрабатывать
TIFF_GLOB  = "*.tif*"       # поймает .tif и .tiff
FITS_GLOB  = "*.fits"

# Авто-стретч по яркости (квантильная обрезка)
P_LOW  = 0.001              # 0.1% снизу
P_HIGH = 0.999              # 99.9% сверху

# Мягкая компрессия динамики (asinh). 0 — отключить
ASINH_ALPHA = 8.0

# Применять sRGB «гамму» после тонмапа
APPLY_SRGB_GAMMA = True
# ===================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def srgb_encode_linear01(x):
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    thresh = 0.0031308
    low = x * 12.92
    high = (1 + a) * np.power(x, 1/2.4) - a
    return np.where(x <= thresh, low, high)

def auto_stretch_luminance(img, p_low=P_LOW, p_high=P_HIGH, asinh_alpha=ASINH_ALPHA):
    """
    Робастный авто-стретч по яркости Y с мягкой компрессией динамики (asinh).
    Сохраняет оттенки: результат = img * (Y_new / (Y+eps)).
    """
    eps = 1e-12

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)

    # клипнем возможные отрицательные «хвосты»
    img = np.maximum(img, 0.0).astype(np.float32)

    # 1) яркость (линейная)
    Y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    # 2) квантили
    lo = np.quantile(Y, p_low)
    hi = np.quantile(Y, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        lo, hi = float(np.min(Y)), float(np.max(Y) + 1e-6)

    # 3) линейный стретч -> [0..1]
    Y_lin = np.clip((Y - lo) / (hi - lo), 0.0, 1.0)

    # 4) asinh-тонмап (мягко вытягивает туманности)
    if asinh_alpha > 0:
        Y_tone = np.arcsinh(asinh_alpha * Y_lin) / np.arcsinh(asinh_alpha)
    else:
        Y_tone = Y_lin

    # 5) переносим тонмап на RGB
    g = Y_tone / (Y + eps)
    out = img * g[..., None]
    return np.clip(out, 0.0, 1.0)

# ---------- Загрузка изображений ----------
def load_tiff_rgb(path):
    arr = imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    # uint16 -> float, линейное [0..1]
    if arr.dtype == np.uint16:
        img = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        img = arr.astype(np.float32) / 255.0
    else:
        img = arr.astype(np.float32)
        # если «сырой» диапазон слишком широкий — ничего, авто-стретч справится
    return img

def load_fits_rgb_color(path):
    """Пытаемся прочитать цветной FITS (HxWx3 или 3xHxW). Возвращаем float32 RGB."""
    data = fits.getdata(path)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3:
        # варианты осей: (H,W,3) или (3,H,W)
        if arr.shape[-1] == 3:
            img = arr
        elif arr.shape[0] == 3:
            img = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f"Не похоже на RGB FITS (shape={arr.shape}): {path}")
    elif arr.ndim == 2:
        # серый -> RGB
        img = np.repeat(arr[..., None], 3, axis=2)
    else:
        raise ValueError(f"Неожиданная размерность FITS: {arr.shape} ({path})")
    return img

def load_fits_rgb_from_planes(r_path, g_path, b_path):
    r = fits.getdata(r_path).astype(np.float32)
    g = fits.getdata(g_path).astype(np.float32)
    b = fits.getdata(b_path).astype(np.float32)
    # выровняем формы
    if r.ndim != 2 or g.ndim != 2 or b.ndim != 2:
        raise ValueError("Моно-FITS каналы должны быть 2D")
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError("Размеры R/G/B FITS не совпадают")
    return np.dstack([r, g, b])

def find_fits_triplets(dirpath):
    """Ищем наборы *_R.fits, *_G.fits, *_B.fits и группируем по базовому имени."""
    fits_files = sorted(glob.glob(os.path.join(dirpath, FITS_GLOB)))
    triplets = []
    # сгруппируем по базовому имени без суффикса _R/_G/_B
    by_base = {}
    for p in fits_files:
        base = os.path.basename(p)
        m = re.match(r"(.+)_([RGB])\.fits$", base, flags=re.IGNORECASE)
        if m:
            root, ch = m.group(1), m.group(2).upper()
            by_base.setdefault(root, {})[ch] = p
    for root, d in by_base.items():
        if all(k in d for k in ("R", "G", "B")):
            triplets.append((root, d["R"], d["G"], d["B"]))
    return triplets

# ---------- Основной конвертер ----------
def save_png(img_float_rgb, out_path_png, apply_srgb=True):
    # тонмап + sRGB
    img_tone = auto_stretch_luminance(img_float_rgb)
    if apply_srgb:
        img_disp = srgb_encode_linear01(img_tone)
    else:
        img_disp = np.clip(img_tone, 0.0, 1.0)
    u8 = (img_disp * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(out_path_png, format="PNG", optimize=True)

def main():
    # ----- TIFF -----
    tiff_paths = sorted(glob.glob(os.path.join(STACK_DIR, TIFF_GLOB)))
    if tiff_paths:
        print(f"TIFF найдено: {len(tiff_paths)}")
        for p in tqdm(tiff_paths, desc="TIFF -> PNG"):
            try:
                img = load_tiff_rgb(p)
                base = os.path.splitext(os.path.basename(p))[0]
                out_png = os.path.join(OUTPUT_DIR, base + ".png")
                save_png(img, out_png, apply_srgb=APPLY_SRGB_GAMMA)
            except Exception as e:
                print(f"[TIFF] Пропуск {p}: {e}")
    else:
        print("TIFF не найдено (это не ошибка).")

    # ----- FITS (цветные) -----
    fits_color = []
    for p in sorted(glob.glob(os.path.join(STACK_DIR, FITS_GLOB))):
        # пропустим явные моно-каналы *_R/_G/_B.fits — их обработаем отдельно
        if re.search(r"_[RGB]\.fits$", p, flags=re.IGNORECASE):
            continue
        fits_color.append(p)

    if fits_color:
        print(f"FITS (цвет) найдено: {len(fits_color)}")
        for p in tqdm(fits_color, desc="FITS color -> PNG"):
            try:
                img = load_fits_rgb_color(p)
                base = os.path.splitext(os.path.basename(p))[0]
                out_png = os.path.join(OUTPUT_DIR, base + ".png")
                save_png(img, out_png, apply_srgb=APPLY_SRGB_GAMMA)
            except Exception as e:
                print(f"[FITS RGB] Пропуск {p}: {e}")
    else:
        print("Цветные FITS не найдены (это не ошибка).")

    # ----- FITS (моно-триплеты R/G/B) -----
    triplets = find_fits_triplets(STACK_DIR)
    if triplets:
        print(f"FITS-триплеты R/G/B: {len(triplets)}")
        for root, r_p, g_p, b_p in tqdm(triplets, desc="FITS R/G/B -> PNG"):
            try:
                img = load_fits_rgb_from_planes(r_p, g_p, b_p)
                out_png = os.path.join(OUTPUT_DIR, root + ".png")
                save_png(img, out_png, apply_srgb=APPLY_SRGB_GAMMA)
            except Exception as e:
                print(f"[FITS R/G/B] Пропуск {root}: {e}")
    else:
        print("FITS-триплеты R/G/B не найдены (это не ошибка).")

    print(f"[OK] PNG предпросмотры в: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
