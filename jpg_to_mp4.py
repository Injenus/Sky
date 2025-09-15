import cv2
import os

def images_to_video(input_dir, output_video_path, fps=10):
    # Сканируем JPG-файлы и сортируем по имени
    images = sorted([
        os.path.join(input_dir, img)
        for img in os.listdir(input_dir)
        if img.lower().endswith('.jpg')
    ])

    if not images:
        print('❌ Не найдено JPG-файлов в указанной папке.')
        return

    # Получаем размеры кадра из первого изображения
    frame = cv2.imread(images[0])
    if frame is None:
        print(f'❌ Невозможно прочитать изображение: {images[0]}')
        return

    height, width, _ = frame.shape

    # Инициализируем видеопишущий объект с 30 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f'⚠️ Пропущено поврежденное изображение: {img_path}')
            continue
        frame = cv2.resize(frame, (width, height))  # Гарантируем одинаковый размер
        video_writer.write(frame)

    video_writer.release()
    print(f'✅ Видео сохранено: {output_video_path}')

# === Указать директории ===
input_directory = r'D:\Avocation\Sky\80. 07-08.09.2025\cropped_1080_edited'
output_video = r'D:\Avocation\Sky\80. 07-08.09.2025\timelapse.mp4'

images_to_video(input_directory, output_video)
