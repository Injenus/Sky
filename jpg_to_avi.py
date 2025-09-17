import cv2
import os

def images_to_uncompressed_avi(input_dir, output_video_path, fps=30):
    images = sorted([
        os.path.join(input_dir, img)
        for img in os.listdir(input_dir)
        if img.lower().endswith('.jpg')
    ])

    if not images:
        print('❌ Нет изображений JPG.')
        return

    frame = cv2.imread(images[0])
    if frame is None:
        print(f'❌ Не удалось прочитать: {images[0]}')
        return

    height, width, _ = frame.shape

    # FOURCC код для несжатого RGB видео (без компрессии)
    fourcc = 0  # raw video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f'⚠️ Пропущено: {img_path}')
            continue
        frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)

    video_writer.release()
    print(f'✅ Несжатое AVI видео сохранено: {output_video_path}')

# === Пути ===
input_directory = r'D:\Avocation\Sky\80. 07-08.09.2025\jpg'
output_video = r'D:\Avocation\Sky\80. 07-08.09.2025\timelapse_rawjpg.avi'

images_to_uncompressed_avi(input_directory, output_video)


