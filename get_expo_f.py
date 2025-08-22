import exifread
# УКАЖИТЕ ПУТЬ К ВАШЕМУ ФАЙЛУ ЗДЕСЬ
file_path = "D:/Avocation/Sky/71. 17.06.2025/raw/valid/DSC06154.ARW"  # ← ИЗМЕНИТЕ ЭТОТ ПУТЬ!

import exifread
import os

def format_exposure(ratio):
    """Форматирует выдержку в виде точной дроби или десятичного значения"""
    if ratio.den == 1:
        return f"{ratio.num} сек"
    elif ratio.num == 1:
        return f"1/{ratio.den} сек"
    else:
        return f"{ratio.num/ratio.den} сек"


# Проверка существования файла
if not os.path.isfile(file_path):
    print(f"Ошибка: Файл не найден: {file_path}")
    exit(1)

try:
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
except Exception as e:
    print(f"Ошибка чтения файла: {e}")
    exit(1)

# Извлечение точной выдержки
exposure_tag = tags.get('EXIF ExposureTime') or tags.get('EXIF ShutterSpeedValue')
if exposure_tag:
    exposure_ratio = exposure_tag.values[0]
    exposure_value = format_exposure(exposure_ratio)
    print(f"Точная выдержка: {exposure_value}")
else:
    print("Информация о выдержке не найдена в EXIF данных")

# Извлечение точной диафрагмы
aperture_tag = tags.get('EXIF FNumber') or tags.get('EXIF ApertureValue')
if aperture_tag:
    aperture_ratio = aperture_tag.values[0]
    aperture_value = float(aperture_ratio.num) / aperture_ratio.den
    print(f"Точная диафрагма: f/{aperture_value}")
else:
    print("Информация о диафрагме не найдена в EXIF данных")