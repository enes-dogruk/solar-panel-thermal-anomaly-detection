#!/usr/bin/env python3
"""
ADIM 4 (video2): PANELLERI KIRP  -- ORIJINAL SURUM (geri alindi)
Video1 ile ayni gerekce (bkz. 4_crop_panels.py basligindaki not).
"""
import cv2, numpy as np, json, csv
from pathlib import Path
from tqdm import tqdm

FRAMES_DIR = Path("1_Extract_frames_v2")
TEMPLATE   = Path("2_panel_data_manual/panels_manual.json")
CROP_DIR   = Path("4_panel_crops_v2")
STATS_CSV  = Path("4_panel_stats_v2.csv")

OUT_SIZE   = 224
SAVE_CROPS = True
PAD        = 2

CROP_DIR.mkdir(exist_ok=True)

panels = json.load(open(TEMPLATE))
frames = sorted(FRAMES_DIR.glob("frame_*.png"))
print(f"{len(panels)} panel sablonu, {len(frames)} kare (video2)")
print(f"Beklenen cikti: {len(panels) * len(frames)} goruntu\n")

rows = []

for f_idx, fpath in enumerate(tqdm(frames, desc="Kirpiliyor")):
    img = cv2.imread(str(fpath))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    frame_mean = float(gray.mean())

    for p in panels:
        pid = p["panel_id"]
        pts = np.array(p["points"], np.int32)

        x, y, w, h = cv2.boundingRect(pts)
        x0, y0 = max(0, x - PAD), max(0, y - PAD)
        x1, y1 = min(W, x + w + PAD), min(H, y + h + PAD)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [pts - [x0, y0]], 255)

        crop_bgr  = img[y0:y1, x0:x1]
        crop_gray = gray[y0:y1, x0:x1]

        px = crop_gray[mask == 255]
        if px.size < 20:
            continue

        rows.append({
            "frame_idx": f_idx,
            "panel_id":  pid,
            "n_px":      int(px.size),
            "mean":      round(float(px.mean()), 2),
            "std":       round(float(px.std()), 2),
            "min":       int(px.min()),
            "max":       int(px.max()),
            "p10":       round(float(np.percentile(px, 10)), 2),
            "p90":       round(float(np.percentile(px, 90)), 2),
            "hot_ratio": round(float((px > px.mean() + 2 * px.std()).mean()), 4),
            "delta_frame": round(float(px.mean() - frame_mean), 2),
            "frame_mean":  round(frame_mean, 2),
        })

        if SAVE_CROPS:
            masked = cv2.bitwise_and(crop_bgr, crop_bgr, mask=mask)
            out = cv2.resize(masked, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(CROP_DIR / f"f{f_idx:04d}_p{pid:03d}.png"), out)

with open(STATS_CSV, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows)

print("\n" + "="*60)
print(f"  Kirpilan panel   : {len(rows)}")
print(f"  Goruntuler       : {CROP_DIR}/")
print(f"  Istatistik CSV   : {STATS_CSV}")
print("="*60)