#!/usr/bin/env python3
"""
ADIM 5 (video2): SICAK LEKE PROFILI  (v3)
Video1 icin gelistirilen hue-tabanli yontemin aynisi, video2 uzerinde.

v2 DUZELTMESI: VERY_BRIGHT esigi eklendi (asiri parlak merkez, doygunluga
  bakmaksizin "sicak" sayilir) - "simit" gorunumunu cozmesi beklendi.

v3 DUZELTMESI (asil sebep buradaymis): hot_mask_check.png onizlemesi
  ref_frame olarak "genel ortalama parlakligi en yuksek" kareyi seciyordu.
  Ama video2'deki arizalar ZAMANLA DEGISKEN (bazi panellerde sadece
  birkac karede zirve yapiyor, cogu karede sonuk) - bu yuzden "en parlak
  genel kare" mutlaka arizalarin en guclu goruldugu an degildi. Metrik
  hesaplamasi (hot_p90) zaten 120 karenin TAMAMINI kullaniyordu ve
  dogruydu; sadece GORSEL ONIZLEME yanlis ana bakiyordu.
  Cozum: ref_frame artik "toplam sicak piksel sayisi EN YUKSEK olan kare"
  olarak seciliyor (tum kareler taranirken bu deger de olculur).
"""
import cv2, numpy as np, json, csv
from pathlib import Path
from tqdm import tqdm
import time

FRAMES_DIR = Path("1_Extract_frames_v2")
TEMPLATE   = Path("2_panel_data_manual/panels_manual.json")
OUT_DIR    = Path("5_hotspot_v2")
OUT_DIR.mkdir(exist_ok=True)

# ---- AYARLAR ----
WARM_HUE_LO  = 0
WARM_HUE_HI  = 35
WARM_WRAP_LO = 165
SAT_MIN      = 55
VAL_MIN      = 70
VERY_BRIGHT  = 150     # bu parlakligin uzeri, doygunluga bakmaksizin "sicak" sayilir

MIN_BLOB     = 0.004
ERODE_MASK   = 2
HOT_THRESH   = 0.01
THUMB        = 130
COLS         = 12

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

panels = json.load(open(TEMPLATE))
frames = sorted(FRAMES_DIR.glob("frame_*.png"))
print(f"{len(panels)} panel, {len(frames)} kare (video2)")
print(f"Yontem: HUE + parlaklik-cekirdek tabanli | siralama: hot_p90\n")

if not frames:
    print("HATA: kare bulunamadi.")
    raise SystemExit(1)

acc = {p["panel_id"]: {"hot": [], "nblob": [], "big": [],
                       "valp95": [], "spread": []} for p in panels}

panel_polys = [np.array(p["points"], np.int32) for p in panels]


def make_mask(pts, x0, y0, x1, y1):
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [pts - [x0, y0]], 255)
    if ERODE_MASK > 0:
        eroded = cv2.erode(mask, K5, iterations=ERODE_MASK)
        if cv2.countNonZero(eroded) >= 20:
            mask = eroded
    return mask


def warm_pixels(hsv_roi, inside):
    """Sicak piksel maskesi: asiri parlak cekirdek OR hue+doygunluk halkasi."""
    h, s, v = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
    warm_hue = ((h >= WARM_HUE_LO) & (h <= WARM_HUE_HI)) | (h >= WARM_WRAP_LO)
    hot_ring = warm_hue & (s >= SAT_MIN) & (v >= VAL_MIN) & inside
    hot_core = (v >= VERY_BRIGHT) & inside
    return hot_core | hot_ring


def panel_metrics(hsv_roi, mask):
    inside = (mask == 255)
    if inside.sum() < 20:
        return None
    area = float(inside.sum())

    hot = warm_pixels(hsv_roi, inside)
    hot_ratio = hot.sum() / area

    n_blob, big_ratio, spread = 0, 0.0, 0.0
    if hot.sum() > 0:
        hm = np.uint8(hot) * 255
        hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, K3)
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

    v = hsv_roi[:, :, 2]
    return {"hot_ratio": hot_ratio, "n_blob": n_blob, "big_ratio": big_ratio,
            "valp95": float(np.percentile(v[inside], 95)), "spread": spread}


# ===== TUM KARELERI TARA (+ referans kare adayini belirle) =====
best_frame = {"path": None, "total_hot": -1}

for fpath in tqdm(frames, desc="Olculuyor"):
    img = cv2.imread(str(fpath))
    if img is None:
        continue
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = hsv_full.shape[:2]

    frame_total_hot = 0

    for p, pts in zip(panels, panel_polys):
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

        frame_total_hot += m["hot_ratio"]

    # Bu kare, simdiye kadarki en cok "toplam ariza gorunurlugu" olan kare mi?
    if frame_total_hot > best_frame["total_hot"]:
        best_frame["total_hot"] = frame_total_hot
        best_frame["path"] = fpath

ref_frame = best_frame["path"]
print(f"\nReferans kare (en cok sicak piksel gosteren an): {ref_frame.name}")
print(f"  (toplam hot_ratio: {best_frame['total_hot']:.2f})")


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
        "label": "",
    })

rows.sort(key=lambda r: -r["hot_p90"])

csv_path = OUT_DIR / "panel_profile.csv"
with open(csv_path, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wtr.writeheader()
    wtr.writerows(rows)


# ===== KONTAK SAYFASI (artik ref_frame = en cok ariza gorunen an) =====
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
    tl = pts[np.argmin(pts.sum(axis=1))]
    cv2.putText(amap, str(r["panel_id"]), (int(tl[0]) + 4, int(tl[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(str(OUT_DIR / "panel_map.png"), amap)


# ===== HOTSPOT MASKE ONIZLEMESI (artik ref_frame = en cok ariza gorunen an) =====
hsv_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
warm = warm_pixels(hsv_ref, np.ones(hsv_ref.shape[:2], bool))
stamp = int(time.time())
mask_filename = f"hot_mask_check_{stamp}.png"
cv2.imwrite(str(OUT_DIR / mask_filename), np.uint8(warm) * 255)
print(f"\n>>> YENI MASKE DOSYASI: {OUT_DIR}/{mask_filename}")


# ===== OZET =====
hp = np.array([r["hot_p90"] for r in rows])
print("\n" + "="*64)
print(f"  Panel sayisi  : {len(rows)}")
print(f"\n  hot_p90 dagilimi:")
for q in [50, 60, 70, 75, 80, 85, 90, 95]:
    print(f"    %{q:<3} : {np.percentile(hp, q):.4f}")
print(f"    max   : {hp.max():.4f}")

print(f"\n  Esige gore panel sayisi:")
for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
    print(f"    hot_p90 > {t:<6}: {(hp > t).sum():>3} panel")

hot_ids = [r["panel_id"] for r in rows if r["hot_p90"] > HOT_THRESH]
print(f"\n  Lekeli paneller (hot_p90 > {HOT_THRESH}): {len(hot_ids)}")
print(f"    {hot_ids}")

print(f"\n  CSV     : {csv_path}")
print(f"  Kontak  : {sheet_path}")
print(f"  Harita  : {OUT_DIR}/panel_map.png")
print(f"  Maske   : {OUT_DIR}/hot_mask_check.png  <- ONCE BUNA BAK")
print("="*64)