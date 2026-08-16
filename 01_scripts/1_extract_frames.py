#!/usr/bin/env python3
# Adım 1 : Videodan Frame çıkarma.

import cv2
from pathlib import Path
from tqdm import tqdm

print("*"*50)
print("VİDEODAN FRAME ÇIKARMA")
print("*"*50)

video_path = "video1.mp4"
output_dir = Path("1_Extract_frames")
output_dir.mkdir(exist_ok = True)

frame_skip = 2 # Her 2. frame'i al

print("Video Dosyası açılıyor.")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("HATA : Video Açılamadı.")
    print(f"Dosya Adı : {video_path}")
    print(f"Konum : Video proje kalsöründe olmalı")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Toplam kaç frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Genişlik kaç piksel
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Yükesklik kaç piksel
duration = frame_count / fps if fps > 0 else 0

print("Video Yüklendi")
print(f"FPS : {fps}")
print(f"Frame Sayısı : {frame_count}")
print(f"Çözünürlük : {width} * {height}")
print(f"Süre : {duration:.1f} saniye\n")

# FRAME ÇIKARMA
print(f"Frameler Çıakrılıyor : (her {frame_skip}. frame).. \n")

extracted_count = 0
frame_number = 0

# Progress bar ile çıkarma
with tqdm(total = frame_count, desc = "Extracting") as pbar:
    while True:
        ret, frame  = cap.read() # Bir frame oku

        if not ret: # video bittiyse çık
            break

        pbar.update(1) # Progress bar güncelle

        # Frame skip (2. Frameleri al)
        if frame_number % frame_skip != 0:
            frame_number += 1
            continue

        # Frame Kaydetme

        output_path = output_dir / f"frame_{extracted_count:05d}.png"
        cv2.imwrite(str(output_path), frame)

        extracted_count += 1
        frame_number += 1

    cap.release()

    # SONUÇLAR
    print("="*50)
    print("Adım 1 Tamamlandı")
    print("="*50)
    print("Sonuçlar:")
    print(f"Toplam Frame : {extracted_count}")
    print(f"Klasör : {output_dir}")
    print(f"Frame Piksel : {width} * {height}")
    print(f"Format :  PNG ")
    print(f"\nFrame Dosya Yolu : {output_dir}/")
    print(f"frame_00000.png (gündüz)")
    print(f"frame_00001.png")
    print(f"frame_00002.png")
    print(f"...")
    print(f"frame_{extracted_count-1:05d}.png")
    print("="*50)
