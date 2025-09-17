#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import json
from datetime import datetime
import numpy as np
import rawpy
from tifffile import imwrite, imread
from astropy.io import fits
from tqdm import tqdm

# ========= ЖЁСТКИЕ ПУТИ/НАСТРОЙКИ =========
RAW_DIR      = r'D:\Avocation\Sky\80. 07-08.09.2025\stack'         # где лежат .ARW
COORDS_DIR   = r'D:\Avocation\Sky\80. 07-08.09.2025\detected\coords'         # где лежат .txt с cx, cy
PREP_DIR     = r'D:\Avocation\Sky\80. 07-08.09.2025\stack\prep_1080'       # куда класть выровненные 1080×1080 TIFF
OUT_DIR      =  r'D:\Avocation\Sky\80. 07-08.09.2025\stack\stack_out'      # итоговые FITS/TIFF
FILE_GLOB    = "*.ARW"

CROP_SIZE    = 1080                               # центральный квадрат
SIGMA        = 3.0                                # sigma-clip
DEMOSAIC     = rawpy.DemosaicAlgorithm.AHD
SAVE_PLANES  = True                               # сохранить моно-FITS R/G/B
# =========================================

os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- разбор координат ----------
coord_re = re.compile(r"cx\s*=\s*([0-9.+-]+)\s*,\s*cy\s*=\s*([0-9.+-]+)", re.IGNORECASE)

def read_coords_for_basename(basename):
    """
    Ожидается файл COORDS_DIR/<basename>.txt с строкой вида:
    cx=2326.500000, cy=1400.500000, R=..., method=...
    Возвращает (cx, cy) как float.
    """
    txt_path = os.path.join(COORDS_DIR, basename + ".txt")
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Нет файла координат: {txt_path}")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    m = coord_re.search(content)
    if not m:
        raise ValueError(f"Не удалось извлечь cx, cy из {txt_path}")
    cx, cy = float(m.group(1)), float(m.group(2))
    return cx, cy

# ---------- чтение RAW -> линейный RGB float ----------
def arw_to_linear_rgb_float(fn):
    with rawpy.imread(fn) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,                # <-- as-shot WB с камеры
            no_auto_bright=True,
            gamma=(1.0, 1.0),                  # линейный выход
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            output_color=rawpy.ColorSpace.sRGB # <-- сразу в sRGB-праймерии
        )
    return (rgb16.astype(np.float32) / 65535.0)


# ---------- паддинги, кроп ----------
def compute_one_side_padding_to_center(W, coord, is_x=True):
    """
    Возвращает пару (pad_before, pad_after) для оси, добавляя паддинг только с одной стороны,
    чтобы объект оказался в центре нового изображения.
    Формула: d = coord - W/2; если d > 0 — добавляем справа (или снизу), если d < 0 — слева (или сверху).
    Величина паддинга = round(|2*d|).
    """
    d = coord - (W / 2.0)
    delta = int(np.round(abs(2.0 * d)))
    if delta <= 0:
        return 0, 0
    if d > 0:
        # объект правее/ниже центра -> добавляем справа/снизу
        return 0, delta
    else:
        # объект левее/выше центра -> добавляем слева/сверху
        return delta, 0

def pad_with_edge(img, pad_left, pad_right, pad_top, pad_bottom):
    return np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")

def ensure_min_size(img, min_w, min_h):
    """
    Если после центрирования ширина/высота меньше требуемых — допаддим по краям «edge»-цветом симметрично.
    """
    H, W, C = img.shape
    need_w = max(0, min_w - W)
    need_h = max(0, min_h - H)
    if need_w == 0 and need_h == 0:
        return img
    left = need_w // 2
    right = need_w - left
    top = need_h // 2
    bottom = need_h - top
    return pad_with_edge(img, left, right, top, bottom)

def crop_center_square(img, size):
    H, W, C = img.shape
    cx = W // 2
    cy = H // 2
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = x0 + size
    y1 = y0 + size
    # На всякий — если вылезли за границы, докинем паддинга «edge»
    add_left = max(0, -x0)
    add_top = max(0, -y0)
    add_right = max(0, x1 - W)
    add_bottom = max(0, y1 - H)
    if any(v > 0 for v in (add_left, add_right, add_top, add_bottom)):
        img = pad_with_edge(img, add_left, add_right, add_top, add_bottom)
        H, W, C = img.shape
        cx = W // 2
        cy = H // 2
        x0 = cx - half
        y0 = cy - half
        x1 = x0 + size
        y1 = y0 + size
    return img[y0:y1, x0:x1, :]

def prepare_all_frames():
    files = sorted(glob.glob(os.path.join(RAW_DIR, FILE_GLOB)))
    if not files:
        files = sorted(glob.glob(os.path.join(RAW_DIR, FILE_GLOB.lower())))
    if not files:
        raise SystemExit(f"Нет .ARW в {RAW_DIR}")

    prepared_paths = []

    for fn in tqdm(files, desc="Подготовка (центрирование+кроп 1080)"):
        base = os.path.splitext(os.path.basename(fn))[0]
        cx, cy = read_coords_for_basename(base)

        # 1) RAW -> линейный RGB float
        img = arw_to_linear_rgb_float(fn)
        H, W, C = img.shape

        # 2) паддинг только с одной стороны на ось X/Y, чтобы объект оказался в центре
        padL, padR = compute_one_side_padding_to_center(W, cx, is_x=True)
        padT, padB = compute_one_side_padding_to_center(H, cy, is_x=False)
        img = pad_with_edge(img, padL, padR, padT, padB)

        # 3) гарантируем, что 1080x1080 поместится
        img = ensure_min_size(img, CROP_SIZE, CROP_SIZE)

        # 4) центральный кроп 1080×1080
        img = crop_center_square(img, CROP_SIZE)

        # 5) сохранить линейный 16-бит TIFF (для стека и контроля)
        out_path = os.path.join(PREP_DIR, base + "_centered_1080.tiff")
        arr16 = (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
        imwrite(out_path, arr16)
        prepared_paths.append(out_path)

    return prepared_paths

# ---------- стек с sigma-clip ----------
def first_pass_mean_std(prep_paths):
    sum_img = None
    sumsq_img = None
    n = 0
    for p in tqdm(prep_paths, desc="Стек: первый проход (mean/std)"):
        img16 = imread(p)  # uint16
        img = img16.astype(np.float32) / 65535.0
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if sum_img is None:
            H, W, C = img.shape
            sum_img   = np.zeros((H, W, C), dtype=np.float64)
            sumsq_img = np.zeros((H, W, C), dtype=np.float64)
        sum_img   += img
        sumsq_img += img * img
        n += 1
    mean = (sum_img / max(n, 1)).astype(np.float32)
    var  = (sumsq_img / max(n, 1) - (mean.astype(np.float64) ** 2)).clip(min=0.0)
    std  = np.sqrt(var).astype(np.float32)
    return mean, std, n

def second_pass_sigma_clip_mean(prep_paths, mean, std, sigma):
    acc = np.zeros_like(mean, dtype=np.float64)
    cnt = np.zeros_like(mean, dtype=np.uint16)
    thr_low  = mean - sigma * std
    thr_high = mean + sigma * std

    for p in tqdm(prep_paths, desc="Стек: второй проход (sigma-clip mean)"):
        img16 = imread(p)
        img = img16.astype(np.float32) / 65535.0
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        mask = (img >= thr_low) & (img <= thr_high)
        acc[mask] += img[mask]
        cnt[mask] += 1

    cnt_safe = np.where(cnt == 0, 1, cnt)
    out = (acc / cnt_safe).astype(np.float32)
    out[cnt == 0] = mean[cnt == 0]
    return out

# ---------- сохранение результатов ----------
def save_fits_color(data_rgb, meta, out_path):
    hdu = fits.PrimaryHDU(data=data_rgb.astype(np.float32))
    hdr = hdu.header
    hdr["BITPIX"]  = -32
    hdr["BUNIT"]   = "relative"
    hdr["NFRAMES"] = meta.get("NFRAMES", 0)
    hdr["SIGMA"]   = meta.get("SIGMA", SIGMA)
    hdr["METHOD"]  = meta.get("METHOD", "SIGCLIP_MEAN")
    hdr["CREATOR"] = "Python rawpy+astropy"
    hdr["DATE"]    = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    for k in ["EXPTIME", "ISO", "FNUMBER", "FOCALLEN", "DATE-OBS", "MAKE", "MODEL"]:
        v = meta.get(k)
        if v is not None:
            try:
                hdr[k] = str(v)
            except Exception:
                pass
    for line in meta.get("HISTORY", []):
        hdr.add_history(line)
    hdu.writeto(out_path, overwrite=True)

def save_fits_planes(data_rgb, meta, base_path):
    names = ["R", "G", "B"]
    for i, nm in enumerate(names):
        mono = data_rgb[..., i].astype(np.float32)
        hdu = fits.PrimaryHDU(data=mono)
        hdr = hdu.header
        hdr["BITPIX"]  = -32
        hdr["BUNIT"]   = "relative"
        hdr["CHANNEL"] = nm
        hdr["NFRAMES"] = meta.get("NFRAMES", 0)
        hdr["SIGMA"]   = meta.get("SIGMA", SIGMA)
        hdr["METHOD"]  = meta.get("METHOD", "SIGCLIP_MEAN")
        hdr["CREATOR"] = "Python rawpy+astropy"
        hdr["DATE"]    = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        for line in meta.get("HISTORY", []):
            hdr.add_history(line)
        out_path = f"{base_path}_{nm}.fits"
        hdu.writeto(out_path, overwrite=True)

def save_tiff16(data_rgb, out_path):
    arr16 = (np.clip(data_rgb, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)
    imwrite(out_path, arr16)

# ---------- EXIF-мета простым способом (по первому кадру) ----------
def read_simple_exif(fn):
    # минимальный лёгкий парсер через rawpy; без внешних зависимостей можно оставить пустым
    meta = {}
    try:
        with rawpy.imread(fn) as raw:
            cam = raw.camera_whitebalance  # просто проверка чтения
            # Можем попробовать извлечь часть тегов
            try:
                meta["MAKE"] = raw.metadata.make
                meta["MODEL"] = raw.metadata.model
                meta["ISO"] = str(raw.metadata.iso_speed)
                meta["EXPTIME"] = str(raw.metadata.shutter)  # строкой
                meta["FNUMBER"] = str(raw.metadata.aperture)
                meta["FOCALLEN"] = str(raw.metadata.focal_len)
            except Exception:
                pass
    except Exception:
        pass
    return meta

# ---------- основной пайплайн ----------
def main():
    print("Шаг 1/3: подготовка кадров (паддинги->центр->кроп 1080)")
    prep_paths = prepare_all_frames()
    if not prep_paths:
        raise SystemExit("Не получилось подготовить кадры.")

    print("Шаг 2/3: стек (sigma-clip mean)")
    mean, std, n = first_pass_mean_std(prep_paths)
    stacked = second_pass_sigma_clip_mean(prep_paths, mean, std, SIGMA)

    print("Шаг 3/3: сохранение результатов")
    # метаданные
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, FILE_GLOB))) or \
                sorted(glob.glob(os.path.join(RAW_DIR, FILE_GLOB.lower())))
    exif_meta = read_simple_exif(raw_files[0]) if raw_files else {}
    meta = {
        "NFRAMES": n,
        "SIGMA": SIGMA,
        "METHOD": "SIGCLIP_MEAN",
        "HISTORY": [
            f"Centered via one-sided edge padding using provided cx,cy; crop {CROP_SIZE}x{CROP_SIZE}",
            f"Stack: sigma-clip mean (sigma={SIGMA})",
            f"Demosaic={DEMOSAIC.name}, linear gamma, no auto-bright, WB=unity",
            f"Prep dir: {PREP_DIR}",
            f"Date: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        ],
        **exif_meta
    }

    # сохраняем
    out_fits_color = os.path.join(OUT_DIR, "stack_1080_linear_color.fits")
    save_fits_color(stacked, meta, out_fits_color)
    print(f"[OK] FITS (RGB): {out_fits_color}")

    if SAVE_PLANES:
        base_planes = os.path.join(OUT_DIR, "stack_1080_linear")
        save_fits_planes(stacked, meta, base_planes)
        print(f"[OK] FITS моно-каналы: {base_planes}_R/G/B.fits")

    out_tiff = os.path.join(OUT_DIR, "stack_1080_linear_16bit.tiff")
    save_tiff16(stacked, out_tiff)
    print(f"[OK] 16-бит TIFF: {out_tiff}")

    # короткий JSON-лог
    info_path = os.path.join(OUT_DIR, "stack_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] Инфо: {info_path}")

if __name__ == "__main__":
    main()
