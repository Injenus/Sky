# -*- coding: utf-8 -*-
# pip install opencv-python numpy scipy
import os
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates

# ================= НАСТРОЙКИ =================
INPUT_DIR   = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\morph")   # <-- входные изображения (маски)
OUTPUT_DIR  = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\detected")# <-- новая директория с результатами
OVERLAYSDir = OUTPUT_DIR / "overlays"                             # <-- сюда кладём размеченные изображения
COORDS_DIR  = OUTPUT_DIR / "coords"                               # <-- сюда кладём txt с координатами
RECURSIVE   = False                                               # искать в подпапках?
OVERWRITE   = True                                                # перезаписывать существующие результаты?
JPEG_QUALITY = 100                                                # качество сохранения оверлеев

# Диапазон радиусов для Хафа (пиксели)
MIN_R = 335
MAX_R = 395

# Параметры Хафа (подстраивайте при необходимости)
HOUGH_DP = 1.0
HOUGH_MIN_DIST = 150
HOUGH_PARAM1 = 120    # верхний порог Canny (нижний возьмётся ~вдвое меньше)
HOUGH_PARAM2 = 1      # порог аккумулятора (меньше -> больше срабатываний)

# Какие расширения считаем изображениями
GLOBS = ("*.jpg", "*.jpeg", "*.png")
# ===============================================================


# ---------- УТИЛИТЫ ----------
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def list_images(root: Path, recursive: bool) -> list[Path]:
    iters = (root.rglob(p) if recursive else root.glob(p) for p in GLOBS)
    files = [p for it in iters for p in it if p.is_file()]
    # Естественная сортировка
    def nkey(s: str):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    return sorted(files, key=lambda p: nkey(str(p)))

def circle_from_3pts(p1, p2, p3):
    (x1,y1), (x2,y2), (x3,y3) = p1, p2, p3
    a = x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2
    if abs(a) < 1e-6:
        return None
    b = ( (x1**2+y1**2)*(y3-y2) + (x2**2+y2**2)*(y1-y3) + (x3**2+y3**2)*(y2-y1) )
    c = ( (x1**2+y1**2)*(x2-x3) + (x2**2+y2**2)*(x3-x1) + (x3**2+y3**2)*(x1-x2) )
    cx = -b/(2*a)
    cy = -c/(2*a)
    R  = np.sqrt((cx-x1)**2 + (cy-y1)**2)
    return cx, cy, R

def kasa_init(points):
    x = points[:,0]; y = points[:,1]
    x_m = x.mean(); y_m = y.mean()
    u = x - x_m; v = y - y_m
    Suu = np.sum(u*u); Svv = np.sum(v*v); Suv = np.sum(u*v)
    Suuu = np.sum(u*u*u); Svvv = np.sum(v*v*v)
    Suvv = np.sum(u*v*v); Svuu = np.sum(v*u*u)
    A = np.array([[Suu, Suv],[Suv, Svv]])
    b = 0.5*np.array([Suuu + Suvv, Svvv + Svuu])
    if np.linalg.cond(A) > 1e12:
        cx, cy = x_m, y_m
    else:
        uc, vc = np.linalg.solve(A, b)
        cx, cy = x_m + uc, y_m + vc
    R = np.mean(np.sqrt((x-cx)**2 + (y-cy)**2))
    return np.array([cx, cy, R], dtype=np.float64)

def geom_residuals(params, pts, weights=None):
    cx, cy, R = params
    r = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2) - R
    return r if weights is None else r * np.sqrt(np.clip(weights, 1e-6, None))

def subpixel_refine_along_normal(gray_f32, pts, gx, gy,
                                 r_out=10.0, r_in=2.0, r_lock=1.2,
                                 step=0.25, guard=0.3):
    refined = []
    H, W = gray_f32.shape
    for (x,y,Gx,Gy) in zip(pts[:,0], pts[:,1], gx, gy):
        g = float(np.hypot(Gx, Gy))
        if g < 1e-6:
            refined.append([x,y]); continue
        nx, ny = Gx/g, Gy/g

        ts_all  = np.arange(-max(r_out, r_in), max(r_out, r_in)+1e-9, step, dtype=np.float32)
        xs_all  = x + ts_all*nx; ys_all = y + ts_all*ny
        I_all = map_coordinates(gray_f32, np.vstack([ys_all, xs_all]),
                                order=1, mode='nearest').astype(np.float32)

        mask_out = (ts_all <= -guard) & (ts_all >= -r_out)
        mask_in  = (ts_all >=  guard) & (ts_all <=  r_in)
        if not np.any(mask_out) or not np.any(mask_in):
            refined.append([x,y]); continue

        I_out = float(np.median(I_all[mask_out]))
        I_in  = float(np.median(I_all[mask_in]))
        if I_in <= I_out:
            refined.append([x,y]); continue

        I_mid = 0.5*(I_out + I_in)
        mask_lock = (ts_all >= -r_lock) & (ts_all <= r_lock)
        T = ts_all[mask_lock]; I = I_all[mask_lock]
        if I.size < 2:
            refined.append([x,y]); continue

        dI = np.diff(I)
        cross = (I[:-1]-I_mid)*(I[1:]-I_mid) <= 0
        if not np.any(cross):
            refined.append([x,y]); continue

        k = np.argmax(np.abs(dI) * cross.astype(np.float32))
        if I[k] == I[k+1]:
            t_sub = T[k]
        else:
            t_sub = T[k] + (I_mid - I[k]) * (T[k+1]-T[k]) / (I[k+1]-I[k])

        xr = float(np.clip(x + t_sub*nx, 0, W-1))
        yr = float(np.clip(y + t_sub*ny, 0, H-1))
        refined.append([xr, yr])

    return np.array(refined, dtype=np.float32)


# ---------- ОСНОВНОЙ ПОИСК КРУГА ДЛЯ ОДНОГО ИЗОБРАЖЕНИЯ ----------
def fit_moon_circle_single(img_gray: np.ndarray) -> tuple | None:
    """
    Принимает 8-битное ЧБ изображение (маску). Возвращает (cx, cy, R) или None.
    """
    # 1) Контраст/шум
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(img_gray)
    g = cv2.GaussianBlur(g, (0,0), sigmaX=1.0)
    g32 = g.astype(np.float32)

    # 2) Кромки и контуры (для набора pts и их субпиксельного уточнения)
    v = np.median(g)
    low = int(max(0, 0.66*v))
    high = int(min(255, 1.33*v))
    edges = cv2.Canny(g, threshold1=low, threshold2=high, L2gradient=True)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=lambda c: len(c), reverse=True)[:10]
    pts = np.vstack([c.reshape(-1,2) for c in cnts]) if cnts else np.empty((0,2), np.int32)
    if len(pts) < 10:
        return None

    pts = pts.astype(np.float32)

    # 3) Субпиксель по нормали
    Gx = cv2.Sobel(g32, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(g32, cv2.CV_32F, 0, 1, ksize=3)
    gx = Gx[pts[:,1].astype(int), pts[:,0].astype(int)]
    gy = Gy[pts[:,1].astype(int), pts[:,0].astype(int)]
    pts_ref = subpixel_refine_along_normal(g32, pts, gx, gy, step=0.25)

    # 4) Хафф в заданном диапазоне радиусов
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=HOUGH_DP, minDist=HOUGH_MIN_DIST,
        #param1=HOUGH_PARAM1, 
        param2=HOUGH_PARAM2,
        minRadius=MIN_R, maxRadius=MAX_R
    )
    if circles is None:
        return None

    cand = circles[0].astype(np.float64)  # (K, 3) -> (x, y, r)

    # Лучшая кандидатура по медианной геом. ошибке относительно pts_ref
    def median_abs_residual(c):
        cx, cy, R = c
        rr = np.sqrt((pts_ref[:,0]-cx)**2 + (pts_ref[:,1]-cy)**2)
        return float(np.median(np.abs(rr - R)))

    best_idx = np.argmin([median_abs_residual(c) for c in cand])
    cx, cy, R = cand[best_idx]
    return float(cx), float(cy), float(R)


def draw_overlay(img_gray: np.ndarray, result: tuple | None) -> np.ndarray:
    """
    Рисуем оверлей: окружность+центр если найдено, иначе только центр кадра.
    """
    H, W = img_gray.shape[:2]
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if result is not None:
        cx, cy, R = result
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(R)), (255, 0, 0), 2)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 21, (0, 0, 255), -1)
        cv2.putText(out, f"cx={cx:.1f} cy={cy:.1f} R={R:.1f}", (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    else:
        cx, cy = (W-1)/2.0, (H-1)/2.0
        cv2.drawMarker(out, (int(round(cx)), int(round(cy))), (0,0,255),
                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(out, f"NO CIRCLE | center=({cx:.1f},{cy:.1f})", (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
    return out


def process_all():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OVERLAYSDir)
    ensure_dir(COORDS_DIR)

    files = list_images(INPUT_DIR, RECURSIVE)
    if not files:
        print(f"[!] Не найдено изображений в {INPUT_DIR}")
        return

    print(f"Найдено файлов: {len(files)}")
    for i, path in enumerate(files, 1):
        try:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[{i}/{len(files)}] Пропуск (не удалось прочитать): {path}")
                continue
            H, W = img.shape[:2]

            # Если результаты уже есть и не хотим перезаписывать — пропускаем
            out_img_path = OVERLAYSDir / (path.stem + "_overlay.jpg")
            out_txt_path = COORDS_DIR   / (path.stem + ".txt")
            if not OVERWRITE and out_img_path.exists() and out_txt_path.exists():
                print(f"[{i}/{len(files)}] Уже существует, пропуск: {path.name}")
                continue

            res = fit_moon_circle_single(img)  # (cx, cy, R) или None

            # Координаты для записи
            if res is not None:
                cx, cy, R = res
                coords_line = f"cx={cx:.6f}, cy={cy:.6f}, R={R:.6f}, method=hough\n"
            else:
                cx, cy = (W-1)/2.0, (H-1)/2.0
                coords_line = f"cx={cx:.6f}, cy={cy:.6f}, R=nan, method=fallback_center\n"

            # Сохраняем coords
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(coords_line)

            # Сохраняем оверлей
            overlay = draw_overlay(img, res)
            cv2.imwrite(str(out_img_path),
                        overlay,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            print(f"[{i}/{len(files)}] OK: {path.name} -> {out_img_path.name}, {out_txt_path.name}")

        except Exception as e:
            # На любой сбой — fallback в центр
            try:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[{i}/{len(files)}] Ошибка чтения и fallback невозможен: {path} :: {e}")
                    continue
                H, W = img.shape[:2]
                cx, cy = (W-1)/2.0, (H-1)/2.0
                out_img_path = OVERLAYSDir / (path.stem + "_overlay.jpg")
                out_txt_path = COORDS_DIR   / (path.stem + ".txt")
                with open(out_txt_path, "w", encoding="utf-8") as f:
                    f.write(f"cx={cx:.6f}, cy={cy:.6f}, R=nan, method=error_fallback\n")
                overlay = draw_overlay(img, None)
                cv2.imwrite(str(out_img_path),
                            overlay,
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                print(f"[{i}/{len(files)}] ERR->FALLBACK center: {path.name} :: {e}")
            except Exception as e2:
                print(f"[{i}/{len(files)}] Фатальная ошибка на {path}: {e} / {e2}")

if __name__ == "__main__":
    process_all()
