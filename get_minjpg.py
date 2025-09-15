import os
import rawpy
import imageio

def extract_thumbnails_from_arw(input_dir, output_dir):
    # Создание выходной директории, если не существует
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.arw'):
            filepath = os.path.join(input_dir, filename)
            try:
                with rawpy.imread(filepath) as raw:
                    thumb = raw.extract_thumb()

                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
                        with open(output_path, 'wb') as f:
                            f.write(thumb.data)
                        print(f'✅ Сохранено превью: {output_path}')
                    else:
                        print(f'⚠️ {filename} содержит неподдерживаемый формат превью: {thumb.format}')
            except Exception as e:
                print(f'❌ Ошибка обработки {filename}: {e}')

# === Указать директории ===
input_directory = r'D:/Avocation/Sky/83. 12-13.09.2025/raw'
output_directory = r'D:/Avocation/Sky/83. 12-13.09.2025/minjpg'

extract_thumbnails_from_arw(input_directory, output_directory)
