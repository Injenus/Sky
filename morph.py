# pip install opencv-python numpy
import cv2
import numpy as np
from pathlib import Path
from itertools import chain
import matplotlib.pyplot as plt
from typing import Optional, Union

# ================= НАСТРОЙКИ =================
INPUT_DIR  = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\jpg")     # <-- исходные JPG
OUTPUT_DIR = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\morph")   # <-- бинарные маски (как было)
OVERLAY_DIR = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\overlay")# <-- НОВОЕ: оверлеи (маска поверх исходника)
HIST_DIR = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\hist")
GRAY_DIR = Path(r"D:\Avocation\Sky\80. 07-08.09.2025\gray")

RECURSIVE = False
OVERWRITE = True
JPEG_QUALITY = 100

HIST_HOT_PIX_THRESHOLD = 150   # порог по яркости (строго > 150)
HIST_HOT_PIX_MIN_COUNT = 42    # минимум пикселей выше порога


MASK_ALPHA = 0.25   # НОВОЕ: прозрачность зелёной маски (0..1, где 1 = полностью зелёная)
GLOBS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
# ============================================

def list_images(root: Path, recursive: bool) -> list[Path]:
    iters = (root.rglob(p) if recursive else root.glob(p) for p in GLOBS)
    files = sorted(set(chain.from_iterable(iters)))
    return [p for p in files if p.is_file()]

def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
        return img
    except Exception:
        return None

def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 100) -> bool:
    try:
        ext = path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 0, 100))]
        else:
            params = []
        ok, buf = cv2.imencode(ext if ext else ".jpg", image, params)
        if not ok:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(path))
        return True
    except Exception:
        return False

# ---------- ВСПОМОГАТЕЛЬНОЕ: зелёный оверлей ----------
def make_green_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Возвращает исходное изображение с полупрозрачной зелёной маской поверх белых пикселей mask.
    alpha — доля "зелени": 0=прозрачно, 1=полностью зелёная заливка в маске.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    else:
        mask = (mask > 0).astype(np.uint8) * 255

    overlay = img_bgr.copy()
    overlay[mask > 0] = (0, 255, 0)  # BGR: зелёный
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0.0)
    return blended

# ========== ТВОЙ АЛГОРИТМ ОБРАБОТКИ =========
def convexify_mask(mask: np.ndarray, per_component: bool = True) -> np.ndarray:
    m = mask.copy()
    if m.dtype != np.uint8:
        m = m.astype(np.uint8)
    m = (m > 0).astype(np.uint8) * 255

    out = np.zeros_like(m)
    if per_component:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            hull = cv2.convexHull(cnt)
            cv2.fillPoly(out, [hull], 255)
    else:
        ys, xs = np.where(m > 0)
        if len(xs) >= 3:
            points = np.column_stack([xs, ys]).astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.fillPoly(out, [hull], 255)
    return out

def sum_and_normalize_channels(img: np.ndarray) -> np.ndarray:
    """
    Суммирует все каналы изображения и нормализует результат в 0..255.
    Возвращает одноканальное изображение uint8.
    """
    # Приводим к float32, чтобы избежать переполнений при суммировании
    if img.ndim == 3:
        summed = img.astype(np.float32).sum(axis=2)  # суммируем ВСЕ каналы (BGR/BGRA и т.д.)
    else:
        # Если уже 1 канал, просто берём его как есть
        summed = img.astype(np.float32)

    # # Нормализация по всему изображению в 0..255
    # minv = float(summed.min())
    # maxv = float(summed.max())
    # if maxv > minv:
    #     out = ((summed - minv) * (255.0 / (maxv - minv))).astype(np.uint8)
    # else:
    #     # Вырожденный случай: изображение константное
    #     out = np.zeros_like(summed, dtype=np.uint8)

    out = np.clip(summed, 0, 255).astype(np.uint8)  # просто обрезаем в 0..255

    return out

def process_image(img_bgr: np.ndarray, hist_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    kernel = np.array([[0,0,1,0,0],
                       [0,1,1,1,0],
                       [1,1,1,1,1],
                       [0,1,1,1,0],
                       [0,0,1,0,0]
                       ], np.uint8)
    total_quantity = img_bgr.shape[0]*img_bgr.shape[1]
    
    #gray_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_image = sum_and_normalize_channels(img_bgr)
    gray_image = cv2.medianBlur(gray_image, 5)
    gray_image = cv2.GaussianBlur(gray_image, (7,7), 0)
    norm = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # # --- Сохранение гистограммы gray_image ---
    # try:
    #     out_path = None
    #     if hist_path is not None:
    #         out_path = Path(hist_path)
    #     else:
    #         # Пытаемся аккуратно вытащить имя исходного файла из кадра вызова (переменная `src`)
    #         import inspect
    #         _f = inspect.currentframe()
    #         _caller = _f.f_back if _f else None
    #         _stem = None
    #         if _caller:
    #             if 'src' in _caller.f_locals:
    #                 _stem = Path(_caller.f_locals['src']).stem
    #             elif 'rel' in _caller.f_locals:
    #                 _stem = Path(_caller.f_locals['rel']).stem
    #         if _stem and 'HIST_DIR' in globals():
    #             out_path = Path(HIST_DIR) / f"{_stem}_hist.jpg"

    #     if out_path is not None:
    #         save_gray_hist_and_log(gray_image, out_path)
    # except Exception:
    #     # не ломаем основной пайплайн обработки в случае любых проблем с сохранением гистограммы
    #     pass
    # # -----------------------------------------

    print(np.max(gray_image), np.min(gray_image), np.mean(gray_image), np.median(gray_image))

    _, img_binary_fixed = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(img_binary_fixed, cv2.MORPH_OPEN, kernel, iterations=3)
    dilation = cv2.dilate(opening, kernel, iterations=1)
    convexify = convexify_mask(dilation, per_component=True)

    hot_count = int(np.count_nonzero(gray_image > HIST_HOT_PIX_THRESHOLD))
    # при недостатке «горячих» пикселей — зануляем маску целиком
    if hot_count < HIST_HOT_PIX_MIN_COUNT:
        #convexify.fill(0)
        pass

    return convexify, gray_image



def save_gray_hist_and_log(image: np.ndarray, output_path: Union[str, Path]) -> Path:
    """
    Строит гистограмму яркостей и ту же гистограмму в лог-масштабе (ось Y),
    размещает их друг под другом и сохраняет в файл.

    Parameters
    ----------
    image : np.ndarray
        2D массив (градации серого). Допустимы dtype uint8/uint16/float.
        Для float предполагается диапазон [0, 1] или [0, 255].
    output_path : str | Path
        Путь к файлу для сохранения (например, 'histograms.png').

    Returns
    -------
    Path
        Фактический путь сохранённого файла.
    """
    # --- проверка входа ---
    if image.ndim != 2:
        raise ValueError("Ожидается 2D изображение (градации серого).")

    # --- приведение к uint8 0..255 для корректной гистограммы ---
    img = image.astype(np.float32, copy=False)
    maxv = float(img.max()) if img.size else 0.0
    if maxv <= 1.0:           # считаем, что картинка в [0,1]
        img = np.clip(img, 0, 1) * 255.0
    else:                      # картинка вероятно уже в [0,255] или шире
        img = np.clip(img, 0, 255)
    img_u8 = img.astype(np.uint8)

    # --- гистограмма ---
    hist, _ = np.histogram(img_u8, bins=256, range=(0, 256))
    bins = np.arange(256)

    # --- рисование ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), dpi=120, sharex=True, constrained_layout=True
    )

    # Обычная гистограмма
    ax1.bar(bins, hist, width=1.0, edgecolor='none')
    ax1.set_title("Гистограмма яркостей")
    ax1.set_ylabel("Количество пикселей")
    ax1.set_xlim(0, 255)
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    # Логарифмическая по Y
    ax2.bar(bins, hist, width=1.0, edgecolor='none')
    ax2.set_title("Гистограмма яркостей (ось Y в лог-масштабе)")
    ax2.set_xlabel("Интенсивность (0..255)")
    ax2.set_ylabel("Количество пикселей (log)")
    ax2.set_xlim(0, 255)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.6, which="both")

    # --- сохранение ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
# =============================================

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Не найдена входная папка: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)   # НОВОЕ
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    GRAY_DIR.mkdir(parents=True, exist_ok=True)

    files = list_images(INPUT_DIR, RECURSIVE)
    if not files:
        print("Нет JPG/JPEG файлов по указанным маскам.")
        return

    total = len(files)
    ok_count = 0
    skip_count = 0
    err_count = 0

    for idx, src in enumerate(files, 1):
        rel = src.relative_to(INPUT_DIR) if RECURSIVE else src.name
        dst_mask = OUTPUT_DIR / rel
        if isinstance(rel, str):
            dst_mask = OUTPUT_DIR / rel
        dst_mask = dst_mask.with_suffix(".jpg")

        dst_overlay = OVERLAY_DIR / (rel if isinstance(rel, str) else rel)
        dst_overlay = dst_overlay.with_suffix(".jpg")

        if not OVERWRITE and (dst_mask.exists() and dst_overlay.exists()):
            skip_count += 1
            print(f"[{idx:>4}/{total}] Пропуск (существуют): {dst_mask} | {dst_overlay}")
            continue

        img = imread_unicode(src)
        if img is None:
            err_count += 1
            print(f"[{idx:>4}/{total}] Ошибка чтения: {src}")
            continue

        try:
            # 1) Считаем маску (как раньше)
            mask, gray = process_image(img)
            if mask is None:
                raise RuntimeError("process_image вернула None")

            if mask.dtype != np.uint8:
                mask = np.clip(mask, 0, 255).astype(np.uint8)

            # 2) НОВОЕ: строим зелёный полупрозрачный оверлей поверх исходника
            overlay_img = make_green_overlay(img, mask, MASK_ALPHA)

            # 3) Сохраняем оба результата
            ok1 = imwrite_unicode(dst_mask, mask, JPEG_QUALITY)
            ok2 = imwrite_unicode(dst_overlay, overlay_img, JPEG_QUALITY)
            ok3 = imwrite_unicode(GRAY_DIR / dst_mask.name, gray, JPEG_QUALITY)

            if ok1 and ok2 and ok3:
                ok_count += 1
                print(f"[{idx:>4}/{total}] OK: {dst_mask} | {dst_overlay}")
            else:
                err_count += 1
                print(f"[{idx:>4}/{total}] Ошибка записи: {dst_mask if not ok1 else ''} {dst_overlay if not ok2 else ''}")

        except Exception as e:
            err_count += 1
            print(f"[{idx:>4}/{total}] Ошибка обработки {src}: {e}")

    print("\nГотово.")
    print(f"Всего файлов: {total}")
    print(f"Успешно    : {ok_count}")
    print(f"Пропущено  : {skip_count} (уже существовали и OVERWRITE=False)")
    print(f"Ошибок     : {err_count}")

if __name__ == "__main__":
    main()
