#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Astro denoise with event protection (meteors/satellites) + FastDVDnet (optional).
Python 3.12+

Пайплайн:
- decode -> YCrCb float
- регистрация (ECC, аффин) по Y (звёзды становятся статичными)
- фон через sigma-clipped median (во времени)
- событийная маска по остатку (сохраняем спутники/метеоры)
- денойз Y (FastDVDnet при наличии; иначе NLM), UV — мягче (Gaussian)
- слияние, кодирование: HEVC 10-bit (NVENC/libx265) или OpenCV 8-bit

Запуск (пример):
  python astro_denoise.py --input noisy.mp4 --output clean.mp4 --window 31 --ky 3.5 ^
      --fastdvdnet_ckpt thirdparty/fastdvdnet/model.pth --device cuda --half
"""

from __future__ import annotations
import argparse, os, sys, shutil, subprocess, io
from collections import deque
from typing import Optional, Tuple, List

import numpy as np
import cv2
from tqdm import tqdm

# -------------------- thirdparty path (FastDVDnet as vendor) --------------------
THIRDPARTY = os.path.join(os.path.dirname(__file__), "thirdparty")
if THIRDPARTY not in sys.path:
    sys.path.insert(0, THIRDPARTY)

# -------------------- utils --------------------

def to_float32(img_bgr_u8: np.ndarray) -> np.ndarray:
    return img_bgr_u8.astype(np.float32) / 255.0

def to_u8_clipped(img_float: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img_float * 255.0), 0, 255).astype(np.uint8)

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def ffmpeg_has_encoder(enc_name: str) -> bool:
    if not has_ffmpeg():
        return False
    try:
        out = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT)
        return enc_name.encode() in out
    except Exception:
        return False

class FFMPEGWriter:
    """Пишем RGB48LE во внешний ffmpeg (NVENC 10-bit, либо libx265 10-bit)."""
    def __init__(self, path: str, w: int, h: int, fps: float, prefer: str = "auto"):
        self.proc = None
        self.w, self.h = w, h
        self.path = path

        if prefer not in ("auto", "nvenc", "x265"):
            prefer = "auto"

        tried = []

        def try_start(cmd: List[str]) -> bool:
            try:
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return True
            except Exception:
                return False

        # Базовая часть
        base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s:v", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
                "-an", "-pix_fmt", "yuv420p10le"]

        nvenc_cmd = base + ["-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", "p5",
                            "-rc", "vbr", "-b:v", "20M", "-maxrate", "40M", "-bf", "3", path]
        x265_cmd  = base + ["-c:v", "libx265", "-x265-params", "profile=main10", path]

        ok = False
        if prefer in ("auto", "nvenc") and ffmpeg_has_encoder("hevc_nvenc"):
            tried.append("hevc_nvenc")
            ok = try_start(nvenc_cmd)
        if not ok and prefer in ("auto", "x265"):
            tried.append("libx265")
            ok = try_start(x265_cmd)

        if not ok:
            raise RuntimeError(f"ffmpeg доступен, но ни один энкодер не стартовал (пробовали: {', '.join(tried)})")

    def write(self, frame_rgb_float: np.ndarray) -> None:
        f16 = np.clip(np.round(frame_rgb_float * 65535.0), 0, 65535).astype(np.uint16)
        self.proc.stdin.write(f16.tobytes(order="C"))

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
                # Прочитаем stderr, чтобы не зависнуть на буфере
                try:
                    _ = self.proc.stderr.read()
                except Exception:
                    pass
                self.proc.wait()
            finally:
                self.proc = None

class CVWriter:
    """Fallback на OpenCV (8-bit BGR)."""
    def __init__(self, path: str, w: int, h: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.vw = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    def write(self, frame_rgb_float: np.ndarray) -> None:
        bgr = cv2.cvtColor(to_u8_clipped(frame_rgb_float), cv2.COLOR_RGB2BGR)
        self.vw.write(bgr)

    def close(self) -> None:
        self.vw.release()

def ecc_register(ref_y: np.ndarray, img_y: np.ndarray,
                 warp_mode: int = cv2.MOTION_AFFINE,
                 number_of_iterations: int = 100,
                 termination_eps: float = 1e-6) -> np.ndarray:
    ref = ref_y.astype(np.float32)
    img = img_y.astype(np.float32)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    try:
        # ECC может падать — оборачиваем
        _, warp_matrix = cv2.findTransformECC(ref, img, warp_matrix, warp_mode, criteria, None, 5)
        return warp_matrix
    except cv2.error:
        return warp_matrix

def apply_warp(img: np.ndarray, W: np.ndarray, warp_mode: int = cv2.MOTION_AFFINE) -> np.ndarray:
    h, w = img.shape[:2]
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(img, W, (w, h),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REFLECT101)
    else:
        return cv2.warpAffine(img, W, (w, h),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT101)

def temporal_mad_sigma(stack: np.ndarray) -> np.ndarray:
    """Оценка sigma по MAD вдоль времени (stack: T,H,W)"""
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0)
    sigma = 1.4826 * mad + 1e-8
    return sigma

def sigma_clip_median(stack: np.ndarray, low: float = 3.0, high: float = 3.0) -> np.ndarray:
    med = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - med), axis=0)
    sigma = 1.4826 * mad + 1e-8
    lo = med - low * sigma
    hi = med + high * sigma
    mask = (stack >= lo) & (stack <= hi)
    clipped = np.where(mask, stack, np.nan)
    bg = np.nanmedian(clipped, axis=0)
    bg = np.where(np.isnan(bg), med, bg)
    return bg

def build_event_mask(y_center: np.ndarray, bg: np.ndarray, sigma: np.ndarray,
                     ky: float = 3.5, min_area: int = 3) -> np.ndarray:
    R = np.abs(y_center - bg)
    M = (R > (ky * sigma)).astype(np.uint8) * 255
    M = cv2.morphologyEx(M, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    if min_area > 1:
        num, labels = cv2.connectedComponents(M, connectivity=8)
        if num > 1:
            areas = np.bincount(labels.ravel())
            kill = np.where(areas < min_area)[0]
            if len(kill) > 0:
                keep = np.ones_like(labels, dtype=bool)
                for k in kill:
                    keep &= (labels != k)
                M = (keep.astype(np.uint8) * 255)
    return M

def nlm_multi_y(window_y_float: List[np.ndarray], center_idx: int,
                h: float = 6.0, template: int = 7, search: int = 21) -> np.ndarray:
    stack_u8 = [to_u8_clipped(y) for y in window_y_float]
    den = cv2.fastNlMeansDenoisingMulti(stack_u8, center_idx, len(stack_u8), None,
                                        h=h, templateWindowSize=template, searchWindowSize=search)
    return den.astype(np.float32) / 255.0

def denoise_uv_per_frame(cr: np.ndarray, cb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cr_d = cv2.GaussianBlur(cr, (0,0), 1.0)
    cb_d = cv2.GaussianBlur(cb, (0,0), 1.0)
    return cr_d, cb_d

# -------------------- FastDVDnet wrapper (FINAL, tensor (1,15,H,W)) --------------------
class FastDVDnetDenoiser:
    """
    Vendor FastDVDnet (thirdparty/fastdvdnet/). Требует PyTorch.
    Веса задайте через --fastdvdnet_ckpt (по умолчанию thirdparty/fastdvdnet/model.pth).
    """
    def __init__(self, ckpt_path: Optional[str], device: str = "cuda", use_half: bool = False):
        self.available = False
        self.model = None
        self.device = device
        self.use_half = use_half
        if not ckpt_path:
            return
        try:
            import torch
            self.torch = torch
            try:
                from fastdvdnet import models as fmodels
                FastDVDnet = getattr(fmodels, "FastDVDnet")
            except Exception:
                from fastdvdnet.models import FastDVDnet  # fallback

            self.model = FastDVDnet(num_input_frames=5)
            state = torch.load(ckpt_path, map_location=device)  # ok: warning от torch.load можно игнорировать
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            self.model.to(device).eval()
            if use_half and device == "cuda":
                self.model.half()
            self.available = True
            print(f"[FastDVDnet] loaded: {ckpt_path}")
        except Exception as e:
            print(f"[FastDVDnet] disabled: {e}", file=sys.stderr)
            self.available = False

    @staticmethod
    def _make_five(window: List[np.ndarray], center_idx: int) -> List[np.ndarray]:
        T = len(window)
        idxs = np.arange(center_idx-2, center_idx+3)
        idxs = np.clip(idxs, 0, T-1)
        return [window[int(i)] for i in idxs]

    def denoise(self, window_y_float: List[np.ndarray], center_idx: int,
                sigma_est: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if not self.available:
            return None

        torch = self.torch
        frames = self._make_five(window_y_float, center_idx)   # list of 5 arrays (H,W) in [0..1]
        H, W = frames[0].shape

        # ---- Сборка входа: (1, 15, H, W) = 5 кадров × 3 канала (RGB=YYY) ----
        x_np = np.stack(frames, axis=0)                        # (5,H,W)
        x_np = np.repeat(x_np[:, None, :, :], 3, axis=1)       # (5,3,H,W)
        x_np = x_np.reshape(1, 5*3, H, W)                      # (1,15,H,W)
        x = torch.from_numpy(x_np).to(self.device)
        if self.use_half and self.device == "cuda":
            x = x.half()
        else:
            x = x.float()

        # ---- Карта шума: (1,1,H,W) ----
        if sigma_est is None:
            sigma_val = 0.03
        else:
            with np.errstate(invalid='ignore'):
                sigma_val = np.nanmedian(sigma_est)
                if not np.isfinite(sigma_val):
                    sigma_val = 0.03
        sigma_val = float(np.clip(sigma_val, 0.005, 0.08))
        noise_map = torch.full((1, 1, H, W), fill_value=sigma_val, device=self.device,
                               dtype=(torch.float16 if (self.use_half and self.device=="cuda") else torch.float32))

        # ---- Вызов модели ----
        out = self.model(x, noise_map)                         # ожидаем (1,3,H,W) либо (1,1,H,W)

        # ---- Приводим к (H,W) ----
        if out.ndim != 4 or out.shape[0] != 1:
            raise RuntimeError(f"Unexpected FastDVDnet output shape: {tuple(out.shape)}")
        # берём 1-й канал (мы подавали YYY, каналы должны быть идентичны)
        out_y = out[:, 0:1, :, :]
        out_y = out_y.squeeze(0).squeeze(0)                    # (H,W)
        out_y = out_y.detach().float().cpu().numpy()
        return np.clip(out_y, 0.0, 1.0).astype(np.float32)


# -------------------- main pipeline --------------------

def write_frame(writer, rgb_out: np.ndarray):
    writer.write(rgb_out)

def process_center_frame(buffer_y: deque, buffer_bgr: deque, half: int,
                         args, fastdvd: Optional[FastDVDnetDenoiser]) -> np.ndarray:
    """Обрабатывает текущий центральный кадр окна и возвращает RGB float [0..1]."""
    win_y: List[np.ndarray] = list(buffer_y)
    y_center = win_y[half]

    # фон + sigma по времени (на зарегистрированном стеке)
    stack = np.stack(win_y, axis=0)  # (T,H,W)
    bg = sigma_clip_median(stack, low=args.clip, high=args.clip)
    sigma = temporal_mad_sigma(stack)

    # событийная маска
    M = build_event_mask(y_center, bg, sigma, ky=args.ky, min_area=args.min_area)  # 0/255
    mask = (M > 0).astype(np.float32)

    # денойз яркости
    if fastdvd is not None and fastdvd.available:
        y_denoised = fastdvd.denoise(win_y, half, sigma_est=sigma)
        if y_denoised is None:
            y_denoised = nlm_multi_y(win_y, half, h=args.nlm_h, template=args.nlm_template, search=args.nlm_search)
    else:
        y_denoised = nlm_multi_y(win_y, half, h=args.nlm_h, template=args.nlm_template, search=args.nlm_search)

    # защита событий
    y_out = y_denoised * (1.0 - mask) + y_center * mask

    # хрома — мягче и покадрово (центр окна)
    bgr_center = list(buffer_bgr)[half]
    ycc_center_u8 = cv2.cvtColor(to_u8_clipped(bgr_center), cv2.COLOR_BGR2YCrCb)
    ycc_center = ycc_center_u8.astype(np.float32) / 255.0
    cr_c, cb_c = ycc_center[:,:,1], ycc_center[:,:,2]
    cr_d, cb_d = denoise_uv_per_frame(cr_c, cb_c)

    # сборка в RGB
    ycc_out = np.stack([np.clip(y_out,0,1), np.clip(cr_d,0,1), np.clip(cb_d,0,1)], axis=2)
    bgr_out_u8 = cv2.cvtColor(to_u8_clipped(ycc_out), cv2.COLOR_YCrCb2BGR)
    bgr_out = bgr_out_u8.astype(np.float32) / 255.0
    rgb_out_u8 = cv2.cvtColor(to_u8_clipped(bgr_out), cv2.COLOR_BGR2RGB)
    rgb_out = rgb_out_u8.astype(np.float32) / 255.0
    return rgb_out

def process_video(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Cannot open input", file=sys.stderr)
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    fps = args.fps if args.fps > 0 else (fps_in if fps_in and fps_in > 0 else 25.0)

    # Writer selection
    writer = None
    if has_ffmpeg():
        try:
            prefer = "auto"
            if args.codec in ("nvenc", "x265"):
                prefer = "nvenc" if args.codec == "nvenc" else "x265"
            writer = FFMPEGWriter(args.output, W, H, fps, prefer=prefer)
            print("Encoding via ffmpeg (10-bit):", "hevc_nvenc" if prefer!="x265" and ffmpeg_has_encoder("hevc_nvenc") else "libx265")
        except Exception as e:
            print(f"ffmpeg writer failed ({e}); falling back to OpenCV (8-bit).")
            writer = CVWriter(args.output, W, H, fps)
    else:
        print("ffmpeg not found; using OpenCV writer (8-bit).")
        writer = CVWriter(args.output, W, H, fps)

    # подготовка окна
    if args.window % 2 == 0 or args.window < 5:
        print("--window must be odd and >=5", file=sys.stderr); sys.exit(2)
    half = args.window // 2
    warp_mode = cv2.MOTION_AFFINE

    # инициализация FastDVDnet
    ckpt_default = os.path.join(THIRDPARTY, "fastdvdnet", "model.pth")
    ckpt_path = args.fastdvdnet_ckpt or (ckpt_default if os.path.isfile(ckpt_default) else "")
    fastdvd = None
    if ckpt_path:
        fastdvd = FastDVDnetDenoiser(ckpt_path, device=args.device, use_half=args.half)

    # читаем первый кадр, чтобы префилл сделать
    ret, first_bgr_u8 = cap.read()
    if not ret:
        print("Empty input.", file=sys.stderr)
        sys.exit(1)

    first_bgr = to_float32(first_bgr_u8)
    ycc0_u8 = cv2.cvtColor(first_bgr_u8, cv2.COLOR_BGR2YCrCb)
    y0 = ycc0_u8[:,:,0].astype(np.float32) / 255.0

    ref_y = y0.copy()
    # буферы
    buffer_bgr: deque = deque(maxlen=args.window)
    buffer_y:   deque = deque(maxlen=args.window)

    # пэддинг начала: кладём половину окна копиями первого кадра
    for _ in range(half):
        buffer_bgr.append(first_bgr)
        buffer_y.append(y0)

    # добавляем сам первый кадр (как реальный)
    buffer_bgr.append(first_bgr)
    buffer_y.append(y0)

    # основной проход
    written = 0
    pbar_total = n_frames if n_frames > 0 else None
    pbar = tqdm(total=pbar_total, desc="Processing", unit="f")

    # как только окно заполняется, начинаем писать
    if len(buffer_y) == args.window:
        rgb_out = process_center_frame(buffer_y, buffer_bgr, half, args, fastdvd)
        write_frame(writer, rgb_out)
        buffer_y.popleft(); buffer_bgr.popleft()
        written += 1

    while True:
        ret, frame_bgr_u8 = cap.read()
        if not ret:
            break

        bgr_f = to_float32(frame_bgr_u8)
        ycc = cv2.cvtColor(frame_bgr_u8, cv2.COLOR_BGR2YCrCb)
        y = ycc[:,:,0].astype(np.float32) / 255.0

        # регистрация текущего Y к ref_y
        Wmat = ecc_register(ref_y, y, warp_mode=warp_mode, number_of_iterations=100)
        y_reg  = apply_warp(y,  Wmat, warp_mode)
        bgr_reg = apply_warp(bgr_f, Wmat, warp_mode)

        buffer_bgr.append(bgr_reg)
        buffer_y.append(y_reg)

        # обновляем ref_y ближе к текущему, чтобы уменьшать дрейф
        ref_y = y_reg

        if len(buffer_y) == args.window:
            rgb_out = process_center_frame(buffer_y, buffer_bgr, half, args, fastdvd)
            write_frame(writer, rgb_out)
            buffer_y.popleft(); buffer_bgr.popleft()
            written += 1

        pbar.update(1)

    cap.release()

    # пэддинг хвоста: добиваем последними кадрами (дублируем последний зарегистрированный)
    if len(buffer_y) > 0:
        last_y = buffer_y[-1]
        last_bgr = buffer_bgr[-1]
        while len(buffer_y) < args.window:
            buffer_y.append(last_y)
            buffer_bgr.append(last_bgr)
        # теперь выжимаем окно до пустоты
        while len(buffer_y) == args.window:
            rgb_out = process_center_frame(buffer_y, buffer_bgr, half, args, fastdvd)
            write_frame(writer, rgb_out)
            buffer_y.popleft(); buffer_bgr.popleft()
            written += 1

    writer.close()
    pbar.close()
    print(f"Done -> {args.output} (frames written: {written})")

def main():
    ap = argparse.ArgumentParser(description="Astro denoise (sigma-clipping + event mask + FastDVDnet optional)")
    ap.add_argument("--input", required=True, help="input video (mp4)")
    ap.add_argument("--output", required=True, help="output video (mp4)")
    ap.add_argument("--window", type=int, default=31, help="temporal window size (odd, >=5)")
    ap.add_argument("--ky", type=float, default=3.5, help="event threshold multiplier for Y")
    ap.add_argument("--clip", type=float, default=3.0, help="sigma-clip for background")
    ap.add_argument("--min_area", type=int, default=3, help="min connected area in event mask (pixels)")
    ap.add_argument("--nlm_h", type=float, default=6.0, help="NLM strength for Y (fallback)")
    ap.add_argument("--nlm_template", type=int, default=7, help="NLM template window size")
    ap.add_argument("--nlm_search", type=int, default=21, help="NLM search window size")
    ap.add_argument("--fps", type=float, default=-1.0, help="override FPS (<=0 to keep source)")
    ap.add_argument("--fastdvdnet_ckpt", type=str, default="", help="path to FastDVDnet .pth (enables GPU denoiser)")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"], help="device for FastDVDnet")
    ap.add_argument("--half", action="store_true", help="use FP16 for FastDVDnet on CUDA")
    ap.add_argument("--codec", type=str, default="auto", choices=["auto","nvenc","x265"],
                    help="prefer ffmpeg encoder: auto (nvenc->x265), nvenc, or x265")
    args = ap.parse_args()

    process_video(args)

if __name__ == "__main__":
    main()
