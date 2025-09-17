"""
Для Windows в консоли запустить:
python путь_до_скрипта\get_metadata.py путь_до_arw желаемый_путь_до_отчётов

для Linux то же самое, но слэши в другую сторону

МОЖНО НЕ УКАЗЫВАТЬ пути, тогда будут использоваться дефолтные
"""

import exifread
import os
import argparse
from datetime import datetime, timedelta

# ДЕФОЛТНЫЕ ПУТИ (можно изменить под свои)
pref = r'D:\Avocation\Sky\83. 12-13.09.2025\raw'
suff = '' #valid
DEFAULT_INPUT_DIR = os.path.join(pref, suff)    # фактический путь <pref>\<suff>
DEFAULT_OUTPUT_DIR = os.path.join(pref, 'metadata') # фактический путь <pref>\metadata

# КОРРЕКЦИЯ ВРЕМЕНИ (в секундах)
# Часы камеры отстают на это количество секунд (может быть больше 60)
TIME_CORRECTION_SECONDS = 70.8
TIME_ZONE = 'UTC+3'
IS_VALID_TIME = False


def format_exposure_from_seconds(sec_value):
    """Форматирует выдержку, заданную в секундах (float), в виде дроби или десятичного значения"""
    if sec_value <= 0:
        return '-'
    # Если секунда целая
    if sec_value.is_integer():
        return f"{int(sec_value)} сек"
    # Если 1 секунда делить на целое
    inv = 1 / sec_value
    if inv.is_integer():
        return f"1/{int(inv)} сек"
    # Попробуем найти простую дробь
    # Ограничим знаменатель
    from fractions import Fraction
    frac = Fraction(sec_value).limit_denominator(1000)
    return f"{frac.numerator}/{frac.denominator} сек"


def format_exposure(ratio, is_apex=False):
    """Форматирует выдержку: если is_apex=True, переводим из APEX в секунды"""
    try:
        # rationals
        val = ratio.num / ratio.den
    except AttributeError:
        val = float(ratio)

    if is_apex:
        # APEX: SSV = log2(1/T) => T = 2^{-SSV}
        sec = 2 ** (-val)
    else:
        sec = val
    return format_exposure_from_seconds(sec)


def format_aperture_from_value(N):
    """Форматирует число диафрагмы (float)"""
    if N <= 0:
        return '-'
    return f"f/{N:.3f}"


def format_aperture(ratio, is_apex=False):
    """Конвертирует апертуру: если is_apex=True, переводим из APEX в число f/"""
    try:
        val = ratio.num / ratio.den
    except AttributeError:
        val = float(ratio)

    if is_apex:
        # AV = 2*log2(N) => N = 2^{AV/2}
        N = 2 ** (val / 2)
    else:
        N = val
    return format_aperture_from_value(N)


def parse_exif_time(main_tag, subsec_tag=None):
    """Парсит EXIF время и возвращает объект datetime или None"""
    if not main_tag:
        return None
        
    try:
        # Базовое время в формате "YYYY:MM:DD HH:MM:SS"
        time_str = str(main_tag).strip()
        if len(time_str) == 19 and time_str[4] == ':' and time_str[7] == ':':
            # Конвертируем в объект datetime
            dt_obj = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
            
            # Добавляем доли секунды если есть
            microseconds = 0
            if subsec_tag:
                subsec_str = str(subsec_tag).strip()
                if subsec_str.isdigit():
                    # Нормализуем до 6 цифр (микросекунды)
                    subsec_str = subsec_str.ljust(6, '0')[:6]
                    microseconds = int(subsec_str)
            
            return dt_obj.replace(microsecond=microseconds)
    except Exception:
        pass
    return None


def apply_time_correction(dt_obj):
    """Применяет коррекцию времени и возвращает исправленный datetime"""
    if not dt_obj:
        return None
    return dt_obj + timedelta(seconds=TIME_CORRECTION_SECONDS)


def format_time_with_rounded_seconds(dt_obj):
    """Форматирует время: YYYY-MM-DD HH:MM:SS.s с округлением секунд до 1 знака"""
    if not dt_obj:
        return "-"
    
    # Округляем микросекунды до десятых секунды
    total_seconds = dt_obj.second + dt_obj.microsecond / 1_000_000
    rounded_seconds = round(total_seconds, 1)
    
    # Обработка случая, когда округление дает 60.0 секунд
    if rounded_seconds >= 60:
        # Переносим лишние секунды в минуты
        extra_minutes = int(rounded_seconds // 60)
        seconds_remainder = rounded_seconds % 60
        dt_obj = dt_obj + timedelta(minutes=extra_minutes)
        rounded_seconds = seconds_remainder
    
    # Форматируем секунды с одним знаком после запятой
    seconds_str = f"{rounded_seconds:04.1f}"  # Всегда с одним знаком после запятой
    
    # Если целая часть секунд < 10, добавляем ведущий ноль
    if float(seconds_str) < 10:
        seconds_str = "0" + seconds_str
    
    return dt_obj.strftime(f"%Y-%m-%d %H:%M:") + seconds_str


def extract_exif_info(file_path):
    """
    Извлекает метаданные из EXIF данных файла.
    Возвращает словарь с ключами: exposure, aperture, iso, camera_time, real_time.
    """
    info = {
        "exposure": "-",
        "aperture": "-",
        "iso": "-",
        "camera_time": "-",
        "real_time": "-"
    }
    
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
    except Exception as e:
        print(f"Ошибка чтения {file_path}: {e}")
        return info

    # Выдержка: предпочитаем прямой тег, иначе APEX
    if 'EXIF ExposureTime' in tags:
        exp_tag = tags['EXIF ExposureTime']
        is_apex = False
    elif 'EXIF ShutterSpeedValue' in tags:
        exp_tag = tags['EXIF ShutterSpeedValue']
        is_apex = True
    else:
        exp_tag = None

    if exp_tag:
        try:
            ratio = exp_tag.values[0]
            info['exposure'] = format_exposure(ratio, is_apex)
        except Exception:
            info['exposure'] = str(exp_tag)

    # Диафрагма: предпочитаем прямой тег, иначе APEX
    if 'EXIF FNumber' in tags:
        aper_tag = tags['EXIF FNumber']
        is_apex = False
    elif 'EXIF ApertureValue' in tags:
        aper_tag = tags['EXIF ApertureValue']
        is_apex = True
    else:
        aper_tag = None

    if aper_tag:
        try:
            ratio = aper_tag.values[0]
            info['aperture'] = format_aperture(ratio, is_apex)
        except Exception:
            info['aperture'] = str(aper_tag)

    # ISO
    iso_tag = tags.get('EXIF ISOSpeedRatings') or tags.get('EXIF PhotographicSensitivity')
    if iso_tag:
        val = iso_tag.values[0] if isinstance(iso_tag.values, (list, tuple)) else iso_tag.values
        info['iso'] = str(val)

    # Извлекаем только оригинальное время съемки
    main_tag = tags.get('EXIF DateTimeOriginal') or tags.get('DateTimeOriginal')
    subsec_tag = tags.get('EXIF SubSecTimeOriginal') or tags.get('SubSecTimeOriginal')
    
    # Парсим время
    parsed_time = parse_exif_time(main_tag, subsec_tag)
    
    # Если нашли время - сохраняем в двух вариантах
    if parsed_time:
        # Форматируем время камеры
        info['camera_time'] = format_time_with_rounded_seconds(parsed_time)
        
        # Применяем коррекцию и форматируем реальное время
        corrected_time = apply_time_correction(parsed_time)
        info['real_time'] = format_time_with_rounded_seconds(corrected_time)

    return info


def generate_reports(input_dir, output_dir):
    """
    Для каждого .ARW файла в input_dir создаем .txt отчет в output_dir
    с метаданными и временем съемки (оригинальным и исправленным).
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Папка не найдена: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Выводим информацию о применяемой коррекции
    correction_minutes = TIME_CORRECTION_SECONDS // 60
    correction_seconds = TIME_CORRECTION_SECONDS % 60
    print(f"Применяем коррекцию времени: +{correction_minutes} мин {correction_seconds} сек")

    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith('.arw'):
                full_path = os.path.join(root, fname)
                info = extract_exif_info(full_path)

                base = os.path.splitext(fname)[0]
                report_path = os.path.join(output_dir, base + '.txt')

                with open(report_path, 'w', encoding='utf-8') as rep:
                    rep.write(f"Файл: {fname}\n")
                    rep.write(f"Выдержка: {info['exposure']}\n")
                    rep.write(f"Диафрагма: {info['aperture']}\n")
                    rep.write(f"ISO: {info['iso']}\n")
                    rep.write(f"Время снимка (камера): {info['camera_time']} {TIME_ZONE}\n")
                    if not IS_VALID_TIME:
                        rep.write(f"Коррекция указана НЕВЕРНО!\n")
                    else:
                        rep.write(f"Время снимка (реальное): {info['real_time']} {TIME_ZONE}\n")
                    rep.write(f"\nПримечание: применена коррекция +{TIME_CORRECTION_SECONDS} сек\n")
                    rep.write(f"Записывается время конца выдержки\n")
                        
                print(f"Сгенерирован отчет: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Генератор EXIF отчетов для ARW файлов')
    parser.add_argument('input_dir', nargs='?', default=DEFAULT_INPUT_DIR,
                        help=f'Путь к папке с ARW файлами (по умолчанию {DEFAULT_INPUT_DIR})')
    parser.add_argument('output_dir', nargs='?', default=DEFAULT_OUTPUT_DIR,
                        help=f'Путь к папке для сохранения отчетов (по умолчанию {DEFAULT_OUTPUT_DIR})')
    args = parser.parse_args()

    try:
        generate_reports(args.input_dir, args.output_dir)
    except Exception as e:
        print(f"Ошибка: {e}")