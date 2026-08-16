#!/usr/bin/env python3
"""
TANI 2: SADECE frame_00061.png'deki (secilen referans kare) V dagilimini olcer.
Bir onceki olcum TUM 120 kareyi taramisti, o yuzden max=255 baska bir karede
olusmus olabilirdi. Bu sefer tek kareye odaklaniyoruz.
"""
import cv2, numpy as np, json
from pathlib import Path

TEMPLATE = Path("2_panel_data_manual/panels_manual.json")
FRAME    = Path("1_Extract_frames_v2/frame_00061.png")

CHECK_IDS = [71, 33, 46, 68, 65, 61, 26, 25]

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}

img = cv2.imread(str(FRAME))
if img is None:
    print(f"HATA: {FRAME} bulunamadi.")
    raise SystemExit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H, W = hsv.shape[:2]

ERODE_MASK = 2
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print(f"{'ID':>4} {'p50_V':>7} {'p90_V':>7} {'p95_V':>7} {'p99_V':>7} {'max_V':>7} {'p90_S':>7}")
print("-"*50)

all_p95 = []
for pid in CHECK_IDS:
    pts = np.array(panels[pid]["points"], np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)

    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [pts - [x0, y0]], 255)
    eroded = cv2.erode(mask, K5, iterations=ERODE_MASK)
    if cv2.countNonZero(eroded) >= 20:
        mask = eroded

    roi = hsv[y0:y1, x0:x1]
    inside = (mask == 255)
    v = roi[:, :, 2][inside]
    s = roi[:, :, 1][inside]

    if v.size == 0:
        print(f"{pid:>4}  VERI YOK")
        continue

    p50, p90, p95, p99 = np.percentile(v, [50, 90, 95, 99])
    s90 = np.percentile(s, 90)
    all_p95.append(p95)
    print(f"{pid:>4} {p50:>7.0f} {p90:>7.0f} {p95:>7.0f} {p99:>7.0f} {v.max():>7.0f} {s90:>7.0f}")

print(f"\np95_V ortalamasi (tum kontrol panelleri): {np.mean(all_p95):.0f}")
print(f"Onerilen VERY_BRIGHT (bu ortalamanin biraz altinda): {int(np.mean(all_p95) * 0.9)}")