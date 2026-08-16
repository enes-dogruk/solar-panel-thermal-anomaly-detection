#!/usr/bin/env python3
"""
ADIM 5: SICAK LEKE PROFILI  (v4)
Her panel icin 121 kare uzerinden birlestirilmis sicak leke metrikleri.

v3: doygunluk yerine HUE tabanli olcum (mor govde elenir)
v4: - ID etiketi poligonun SOL UST kosesine yazilir (merkez yaniltiyordu)
    - hot_p90 kolonu eklendi (zayif ama gercek lekeler medyanla kayboluyordu)
    - siralama hot_p90'a gore
"""
import cv2, numpy as np, json, csv
from pathlib import Path
from tqdm import tqdm

FRAMES_DIR = Path("1_Extract_frames")
TEMPLATE   = Path("2_panel_data_manual/panels_manual.json")
OUT_DIR    = Path("5_hotspot")
OUT_DIR.mkdir(exist_ok=True)

# ---- AYARLAR ----
WARM_HUE_LO  = 0       # turuncu-sari alt sinir (OpenCV hue: 0-180)
WARM_HUE_HI  = 35
WARM_WRAP_LO = 165     # kirmizi ust taraftan sariyor
SAT_MIN      = 55      # bunun altindaki doygunluk = gri, yok say
VAL_MIN      = 70      # cok karanlik pikselleri yok say

MIN_BLOB     = 0.004   # panel alaninin %0.4'unden kucuk lekeler gurultu
ERODE_MASK   = 2       # maskeyi iceri cek (komsu panel sizintisi icin)
HOT_THRESH   = 0.01    # haritada kirmizi isaretleme esigi (hot_p90 uzerinden)
THUMB        = 130
COLS         = 12

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

panels = json.load(open(TEMPLATE))
frames = sorted(FRAMES_DIR.glob("frame_*.png"))
print(f"{len(panels)} panel, {len(frames)} kare")
print(f"Yontem: HUE tabanli | siralama: hot_p90\n")

if not frames:
    print("HATA: kare bulunamadi.")
    raise SystemExit(1)

acc = {p["panel_id"]: {"hot": [], "nblob": [], "big": [],
                       "valp95": [], "spread": []} for p in panels}

bright = [(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).mean(), f) for f in frames]
ref_frame = max(bright, key=lambda x: x[0])[1]


def make_mask(pts, x0, y0, x1, y1):
    """Poligon maskesi + kenar sizintisini onlemek icin erode."""
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [pts - [x0, y0]], 255)
    if ERODE_MASK > 0:
        eroded = cv2.erode(mask, K5, iterations=ERODE_MASK)
        if cv2.countNonZero(eroded) >= 20:
            mask = eroded
    return mask


def panel_metrics(hsv_roi, mask):
    """Sicak leke = TURUNCU/SARI hue. Mor govde ve gri cati elenir."""
    h, s, v = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
    inside = (mask == 255)
    if inside.sum() < 20:
        return None
    area = float(inside.sum())

    warm_hue = ((h >= WARM_HUE_LO) & (h <= WARM_HUE_HI)) | (h >= WARM_WRAP_LO)
    hot = warm_hue & (s >= SAT_MIN) & (v >= VAL_MIN) & inside
    hot_ratio = hot.sum() / area

    n_blob, big_ratio, spread = 0, 0.0, 0.0
    if hot.sum() > 0:
        hm = np.uint8(hot) * 255
        hm = cv2.morphologyEx(hm, cv2.MORPH_OPEN, K3)
        cnts, _ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sizes = [cv2.contourArea(c) for c in cnts]
        sizes = [x for x in sizes if x / area >= MIN_BLOB]
        n_blob = len(sizes)
        if sizes:
            big_ratio = max(sizes) / area
            ys, xs = np.nonzero(hm)
            if xs.size > 1:
                spread = float(np.sqrt(xs.var() + ys.var()) / np.sqrt(area))

    return {"hot_ratio": hot_ratio, "n_blob": n_blob, "big_ratio": big_ratio,
            "valp95": float(np.percentile(v[inside], 95)), "spread": spread}


# ===== TUM KARELERI TARA =====
for fpath in tqdm(frames, desc="Olculuyor"):
    img = cv2.imread(str(fpath))
    if img is None:
        continue
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = hsv_full.shape[:2]

    for p in panels:
        pts = np.array(p["points"], np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        mask = make_mask(pts, x0, y0, x1, y1)
        m = panel_metrics(hsv_full[y0:y1, x0:x1], mask)
        if m is None:
            continue

        a = acc[p["panel_id"]]
        a["hot"].append(m["hot_ratio"])
        a["nblob"].append(m["n_blob"])
        a["big"].append(m["big_ratio"])
        a["valp95"].append(m["valp95"])
        a["spread"].append(m["spread"])


# ===== PANEL BAZINDA OZETLE =====
rows = []
for p in panels:
    pid = p["panel_id"]
    a = acc[pid]
    if not a["hot"]:
        continue
    hot = np.array(a["hot"])
    pts = np.array(p["points"], np.int32)
    cx, cy = pts.mean(axis=0)
    rows.append({
        "panel_id": pid,
        "cx": int(cx), "cy": int(cy),
        "hot_p90":    round(float(np.percentile(hot, 90)), 4),
        "hot_med":    round(float(np.median(hot)), 4),
        "hot_max":    round(float(hot.max()), 4),
        "hot_stable": round(float((hot > 0.005).mean()), 3),
        "nblob_med":  round(float(np.median(a["nblob"])), 2),
        "big_med":    round(float(np.median(a["big"])), 4),
        "valp95_med": round(float(np.median(a["valp95"])), 1),
        "spread_med": round(float(np.median(a["spread"])), 3),
        "label": "",          # SEN DOLDURACAKSIN
    })

rows.sort(key=lambda r: -r["hot_p90"])

csv_path = OUT_DIR / "panel_profile.csv"
with open(csv_path, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows)


# ===== KONTAK SAYFASI =====
ref = cv2.imread(str(ref_frame))
Hf, Wf = ref.shape[:2]

tiles = []
for r in rows:
    p = next(q for q in panels if q["panel_id"] == r["panel_id"])
    pts = np.array(p["points"], np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(Wf, x + w), min(Hf, y + h)

    mask = make_mask(pts, x0, y0, x1, y1)
    crop = cv2.bitwise_and(ref[y0:y1, x0:x1], ref[y0:y1, x0:x1], mask=mask)
    t = cv2.resize(crop, (THUMB, THUMB), interpolation=cv2.INTER_CUBIC)

    cv2.rectangle(t, (0, 0), (THUMB - 1, 24), (0, 0, 0), -1)
    cv2.putText(t, f"{r['panel_id']}", (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(t, f"p{r['hot_p90']:.3f} b{int(r['nblob_med'])}",
                (34, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(t, (0, 0), (THUMB - 1, THUMB - 1), (60, 60, 60), 1)
    tiles.append(t)

nrow = (len(tiles) + COLS - 1) // COLS
sheet = np.zeros((nrow * THUMB, COLS * THUMB, 3), np.uint8)
for i, t in enumerate(tiles):
    rr, cc = divmod(i, COLS)
    sheet[rr*THUMB:(rr+1)*THUMB, cc*THUMB:(cc+1)*THUMB] = t

sheet_path = OUT_DIR / "contact_sheet.png"
cv2.imwrite(str(sheet_path), sheet)


# ===== PANEL HARITASI =====
amap = ref.copy()
for r in rows:
    p = next(q for q in panels if q["panel_id"] == r["panel_id"])
    pts = np.array(p["points"], np.int32)
    col = (0, 0, 255) if r["hot_p90"] > HOT_THRESH else (0, 255, 0)
    cv2.polylines(amap, [pts], True, col, 2)

    # ID etiketi: poligonun SOL UST kosesine (merkez komsuya kayabiliyordu)
    tl = pts[np.argmin(pts.sum(axis=1))]
    cv2.putText(amap, str(r["panel_id"]), (int(tl[0]) + 4, int(tl[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

cv2.imwrite(str(OUT_DIR / "panel_map.png"), amap)


# ===== HOTSPOT MASKE ONIZLEMESI =====
hsv_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
hr, sr, vr = hsv_ref[:, :, 0], hsv_ref[:, :, 1], hsv_ref[:, :, 2]
warm = (((hr >= WARM_HUE_LO) & (hr <= WARM_HUE_HI)) | (hr >= WARM_WRAP_LO)) \
       & (sr >= SAT_MIN) & (vr >= VAL_MIN)
cv2.imwrite(str(OUT_DIR / "hot_mask_check.png"), np.uint8(warm) * 255)


# ===== OZET =====
hp = np.array([r["hot_p90"] for r in rows])
print("\n" + "="*64)
print(f"  Panel sayisi  : {len(rows)}")
print(f"  Referans kare : {ref_frame.name}")
print(f"\n  hot_p90 dagilimi:")
for q in [50, 60, 70, 75, 80, 85, 90, 95]:
    print(f"    %{q:<3} : {np.percentile(hp, q):.4f}")
print(f"    max   : {hp.max():.4f}")

print(f"\n  Esige gore panel sayisi:")
for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
    print(f"    hot_p90 > {t:<6}: {(hp > t).sum():>3} panel")

print(f"\n  Lekeli paneller (hot_p90 > {HOT_THRESH}):")
hot_ids = [r["panel_id"] for r in rows if r["hot_p90"] > HOT_THRESH]
print(f"    {hot_ids}")
print(f"    Toplam: {len(hot_ids)} / {len(rows)}")

print(f"\n  CSV     : {csv_path}")
print(f"  Kontak  : {sheet_path}")
print(f"  Harita  : {OUT_DIR}/panel_map.png")
print(f"  Maske   : {OUT_DIR}/hot_mask_check.png")
print("="*64)