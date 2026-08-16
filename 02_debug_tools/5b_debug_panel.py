#!/usr/bin/env python3
"""Belirli panellerin kare kare hot_ratio degerini gosterir."""
import cv2, numpy as np, json
from pathlib import Path

CHECK_IDS = [1, 0, 2, 5, 7, 72, 15]      # bakmak istedigin panel id'leri

WARM_HUE_HI, WARM_WRAP_LO = 35, 165
SAT_MIN, VAL_MIN, ERODE_MASK = 55, 70, 2
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

panels = {p["panel_id"]: p for p in json.load(open("2_panel_data_manual/panels_manual.json"))}
frames = sorted(Path("1_Extract_frames").glob("frame_*.png"))
Path("5_hotspot/debug").mkdir(parents=True, exist_ok=True)

series = {i: [] for i in CHECK_IDS}
best = {i: (-1, None, None) for i in CHECK_IDS}

for fi, f in enumerate(frames):
    img = cv2.imread(str(f))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = hsv.shape[:2]

    for pid in CHECK_IDS:
        if pid not in panels:
            continue
        pts = np.array(panels[pid]["points"], np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x0, y0, x1, y1 = max(0,x), max(0,y), min(W,x+w), min(H,y+h)
        if x1-x0 < 8 or y1-y0 < 8:
            continue

        m = np.zeros((y1-y0, x1-x0), np.uint8)
        cv2.fillPoly(m, [pts - [x0, y0]], 255)
        er = cv2.erode(m, K5, iterations=ERODE_MASK)
        if cv2.countNonZero(er) >= 20:
            m = er

        roi = hsv[y0:y1, x0:x1]
        hh, ss, vv = roi[:,:,0], roi[:,:,1], roi[:,:,2]
        warm = ((hh <= WARM_HUE_HI) | (hh >= WARM_WRAP_LO)) & (ss>=SAT_MIN) & (vv>=VAL_MIN)
        hot = warm & (m == 255)
        ratio = hot.sum() / max(int((m==255).sum()), 1)
        series[pid].append(ratio)

        if ratio > best[pid][0]:
            best[pid] = (ratio, fi, (img[y0:y1, x0:x1].copy(), m.copy(), np.uint8(hot)*255))

print(f"{'ID':>4} {'med':>8} {'p75':>8} {'p90':>8} {'max':>8} {'>0 kare':>9}")
print("-"*52)
for pid in CHECK_IDS:
    a = np.array(series[pid])
    if a.size == 0:
        print(f"{pid:>4}  VERI YOK"); continue
    print(f"{pid:>4} {np.median(a):>8.4f} {np.percentile(a,75):>8.4f} "
          f"{np.percentile(a,90):>8.4f} {a.max():>8.4f} {(a>0).sum():>6}/{a.size}")

    r, fi, imgs = best[pid]
    if imgs:
        crop, mask, hotm = imgs
        crop_m = cv2.bitwise_and(crop, crop, mask=mask)
        trio = np.hstack([crop, crop_m, cv2.cvtColor(hotm, cv2.COLOR_GRAY2BGR)])
        trio = cv2.resize(trio, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"5_hotspot/debug/panel_{pid:03d}_f{fi:04d}.png", trio)

print("\nGoruntuler -> 5_hotspot/debug/")
print("Her dosyada 3 panel: [ham kirpim] [maskeli] [sicak maske]")