import subprocess
import shutil
import sys
from pathlib import Path

# ===== ЖЁСТКИЕ ПУТИ И НАСТРОЙКИ =====
INPUT_MP4  = Path(r'D:\Avocation\Sky\80. 07-08.09.2025\timelapse.mp4')               # исходный mp4
OUTPUT_MP4 = Path(r'D:\Avocation\Sky\80. 07-08.09.2025\timelapse_audio.mp4')   # результат mp4
FFMPEG_BIN = "ffmpeg"     # если нужно, пропиши полный путь, например: "/usr/bin/ffmpeg" или "C:\\ffmpeg\\bin\\ffmpeg.exe"
FFPROBE_BIN = "ffprobe"   # аналогично, при необходимости укажи полный путь

# Настройки бесшумного звука
AUDIO_SR = 48000          # частота дискретизации
AUDIO_CHANNELS = 2        # 1=mono, 2=stereo
AUDIO_BITRATE = "8k"      # очень низкий битрейт, чтобы «дёшево» занимало место
FASTSTART = True          # переместить moov в начало (удобно для веб-плееров)

# ====================================

def ensure_tools():
    if not shutil.which(FFMPEG_BIN):
        sys.stderr.write(f"Ошибка: ffmpeg не найден ({FFMPEG_BIN}). Установи ffmpeg и/или пропиши полный путь в FFMPEG_BIN.\n")
        sys.exit(1)
    if not shutil.which(FFPROBE_BIN):
        sys.stderr.write(f"Ошибка: ffprobe не найден ({FFPROBE_BIN}). Установи ffmpeg (в комплекте есть ffprobe) и/или пропиши путь в FFPROBE_BIN.\n")
        sys.exit(1)

def has_audio_stream(video: Path) -> bool:
    # Проверяем наличие аудиопотока через ffprobe
    # Если найдёт хотя бы один аудиопоток — вернёт True
    try:
        res = subprocess.run(
            [
                FFPROBE_BIN, "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(video),
            ],
            capture_output=True, text=True, check=False
        )
        return res.stdout.strip() != ""
    except Exception:
        return False

def copy_container(src: Path, dst: Path):
    cmd = [FFMPEG_BIN, "-y", "-i", str(src), "-c", "copy"]
    if FASTSTART:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(dst)]
    subprocess.run(cmd, check=True)

def add_silent_audio(src: Path, dst: Path):
    ch_layout = "mono" if AUDIO_CHANNELS == 1 else "stereo"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(src),
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout={ch_layout}:sample_rate={AUDIO_SR}",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-shortest",
    ]
    if FASTSTART:
        cmd += ["-movflags", "+faststart"]
    cmd += [str(dst)]
    subprocess.run(cmd, check=True)

def main():
    ensure_tools()

    if not INPUT_MP4.exists():
        sys.stderr.write(f"Ошибка: входной файл не найден: {INPUT_MP4}\n")
        sys.exit(1)

    OUTPUT_MP4.parent.mkdir(parents=True, exist_ok=True)

    try:
        if has_audio_stream(INPUT_MP4):
            print("У входного видео уже есть аудиодорожка. Просто копирую контейнер без перекодирования...")
            copy_container(INPUT_MP4, OUTPUT_MP4)
        else:
            print("Аудиодорожка отсутствует. Добавляю бесшумную аудиодорожку (AAC)...")
            add_silent_audio(INPUT_MP4, OUTPUT_MP4)
        print(f"Готово: {OUTPUT_MP4}")
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"ffmpeg/ffprobe завершился с ошибкой (код {e.returncode}). Проверь входной файл и права доступа.\n")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
