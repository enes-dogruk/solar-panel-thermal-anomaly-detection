#!/usr/bin/env python3
# Adım 1 (video2): Videodan Frame çıkarma.

import cv2
from pathlib import Path
from tqdm import tqdm

print("*"*50)
print("VİDEODAN FRAME ÇIKARMA (video2)")
print("*"*50)

video_path = "video2.mp4"
output_dir = Path("1_Extract_frames_v2")
output_dir.mkdir(exist_ok=True)

frame_skip = 2

print("Video Dosyası açılıyor.")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("HATA : Video Açılamadı.")
    print(f"Dosya Adı : {video_path}")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0

print(f"FPS : {fps}")
print(f"Frame Sayısı : {frame_count}")
print(f"Çözünürlük : {width} * {height}")
print(f"Süre : {duration:.1f} saniye\n")

extracted_count = 0
frame_number = 0

with tqdm(total=frame_count, desc="Extracting") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        if frame_number % frame_skip != 0:
            frame_number += 1
            continue

        output_path = output_dir / f"frame_{extracted_count:05d}.png"
        cv2.imwrite(str(output_path), frame)

        extracted_count += 1
        frame_number += 1

cap.release()

print("="*50)
print(f"Toplam Frame : {extracted_count}")
print(f"Klasör : {output_dir}")
print("="*50)