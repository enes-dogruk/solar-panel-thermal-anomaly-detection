#!/usr/bin/env python3
"""
TANI: En sicak panellerin gercek V (parlaklik) dagilimini olcer.
VERY_BRIGHT esigini tahmin yerine olcume dayali secmek icin.
"""
import cv2, numpy as np, json
from pathlib import Path

TEMPLATE = Path("2_panel_data_manual/panels_manual.json")
FRAMES   = Path("1_Extract_frames_v2")

# Kontak sayfasinda en yuksek hot_p90'a sahip birkac panel
CHECK_IDS = [71, 33, 46, 68, 65, 61]

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}
frames = sorted(FRAMES.glob("frame_*.png"))

ERODE_MASK = 2
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Her panel icin TUM karelerdeki en yuksek V degerlerini topla
all_max_v = {pid: [] for pid in CHECK_IDS}
all_p99_v = {pid: [] for pid in CHECK_IDS}

for fpath in frames:
    img = cv2.imread(str(fpath))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]

    for pid in CHECK_IDS:
        pts = np.array(panels[pid]["points"], np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [pts - [x0, y0]], 255)
        eroded = cv2.erode(mask, K5, iterations=ERODE_MASK)
        if cv2.countNonZero(eroded) >= 20:
            mask = eroded

        v = hsv[y0:y1, x0:x1, 2]
        inside = (mask == 255)
        vals = v[inside]
        if vals.size == 0:
            continue
        all_max_v[pid].append(int(vals.max()))
        all_p99_v[pid].append(float(np.percentile(vals, 99)))

print(f"{'ID':>4} {'max_V (tum karelerin en yuksegi)':>32} {'p99_V medyani':>16}")
print("-"*56)
for pid in CHECK_IDS:
    if not all_max_v[pid]:
        print(f"{pid:>4}  VERI YOK")
        continue
    overall_max = max(all_max_v[pid])
    p99_med = np.median(all_p99_v[pid])
    print(f"{pid:>4} {overall_max:>32} {p99_med:>16.1f}")

overall = [v for pid in CHECK_IDS for v in all_max_v[pid]]
print(f"\nTum kontrol panellerinin genel maksimumu: {max(overall) if overall else 'N/A'}")
print(f"Onerilen VERY_BRIGHT (genel max'in ~%85'i): "
      f"{int(max(overall)*0.85) if overall else 'N/A'}")