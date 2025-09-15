# strip_audio_hardcoded.py
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# === ЖЁСТКИЕ ПУТИ — ПОМЕНЯЙ ПОД СЕБЯ ===
INPUT_PATH  = r"D:\Avocation\Sky\noisy.mp4"      # исходный файл
OUTPUT_PATH = r"D:\Avocation\Sky\noisy_mute.mp4"  # куда сохранить без звука
INPLACE = False  # True — перезаписать исходный файл; False — писать в OUTPUT_PATH
# =======================================

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg не найден в PATH. Установи и добавь в PATH (Linux: sudo apt install ffmpeg; "
            "macOS: brew install ffmpeg; Windows: choco install ffmpeg или добавь bin в PATH)."
        )

def drop_audio_copy_streams(inp: str, outp: str):
    """
    Убираем все аудиодорожки, при этом копируем остальные потоки (видео, субтитры, таймкоды) без перекодирования.
    -map 0      — брать все потоки из исходника
    -map -0:a   — исключить все аудиопотоки
    -c copy     — копировать без перекодирования (качество не меняется)
    -movflags +faststart — удобно для веб-плееров (moov в начало)
    -map_metadata 0 — сохранить метаданные файла
    """
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-y",                 # перезаписывать выходной файл, если существует
        "-i", inp,
        "-map", "0",
        "-map", "-0:a",
        "-c", "copy",
        "-map_metadata", "0",
        "-movflags", "+faststart",
        outp,
    ]
    subprocess.check_call(cmd)

def main():
    ensure_ffmpeg()

    inp = Path(INPUT_PATH)
    if not inp.is_file():
        raise FileNotFoundError(f"Файл не найден: {inp}")

    if INPLACE:
        # Записываем во временный файл рядом с исходником и атомарно заменяем
        tmp_dir = inp.parent
        with tempfile.NamedTemporaryFile(prefix="muted_", suffix=inp.suffix, dir=tmp_dir, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            drop_audio_copy_streams(str(inp), str(tmp_path))
            os.replace(tmp_path, inp)  # атомарная замена
            print(f"Готово (перезаписан исходник без звука): {inp}")
        finally:
            # На случай сбоев — подчистим временный файл
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    else:
        outp = Path(OUTPUT_PATH)
        outp.parent.mkdir(parents=True, exist_ok=True)
        drop_audio_copy_streams(str(inp), str(outp))
        print(f"Готово (создан новый файл без звука): {outp}")

if __name__ == "__main__":
    main()
