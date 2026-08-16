#!/usr/bin/env python3
"""
TESHIS: Belirli bir kumenin (varsayilan K1) komsuluk kenarlarini gosterir.
Her kenarin mesafe/esik oranini yazdirir ve kumeyi harita uzerinde
kendi kenar baglantilariyla birlikte vurgular - boylece kume gercekten
fiziksel olarak butun mu, yoksa zincirleme bir yanlis birlesme mi
oldugunu gozle (ve sayisal olarak) dogrulayabiliriz.
"""
import cv2, numpy as np, json, csv
from pathlib import Path

TARGET_CLUSTER_PANELS = [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 21, 61]  # K1

PROFILE  = Path("5_hotspot_v2/panel_profile.csv")
TEMPLATE = Path("2_panel_data_manual/panels_manual.json")
FRAMES   = Path("1_Extract_frames_v2")

NEIGH_FACTOR = 1.35

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}
rows = list(csv.DictReader(open(PROFILE)))
for r in rows:
    r["panel_id"] = int(r["panel_id"])
    r["cx"], r["cy"] = int(r["cx"]), int(r["cy"])

all_centers = {r["panel_id"]: (r["cx"], r["cy"]) for r in rows}

def nn_dist(pid):
    cx, cy = all_centers[pid]
    d = [np.hypot(cx-ocx, cy-ocy) for opid,(ocx,ocy) in all_centers.items() if opid != pid]
    return min(d)

nn = {pid: nn_dist(pid) for pid in TARGET_CLUSTER_PANELS}

print(f"Kume panelleri: {TARGET_CLUSTER_PANELS}\n")
print(f"{'panel':>6} {'nn_dist':>9}")
print("-"*18)
for pid in TARGET_CLUSTER_PANELS:
    print(f"{pid:>6} {nn[pid]:>9.1f}")

print(f"\nKUME ICI KOMSULUK KENARLARI (are_neighbors=True olanlar):")
print(f"{'kenar':>10} {'mesafe':>8} {'esik':>8} {'oran':>6}")
print("-"*36)

edges_found = []
for i in range(len(TARGET_CLUSTER_PANELS)):
    for j in range(i+1, len(TARGET_CLUSTER_PANELS)):
        a, b = TARGET_CLUSTER_PANELS[i], TARGET_CLUSTER_PANELS[j]
        cx1, cy1 = all_centers[a]; cx2, cy2 = all_centers[b]
        d = np.hypot(cx1-cx2, cy1-cy2)
        lim = (nn[a] + nn[b]) / 2 * NEIGH_FACTOR
        if d < lim:
            edges_found.append((a, b, d, lim))
            print(f"{a:>4}-{b:<4} {d:>8.1f} {lim:>8.1f} {d/lim:>6.2f}")

print(f"\nToplam kenar sayisi: {len(edges_found)}")

# ===== GORSEL =====
frame_files = sorted(FRAMES.glob("frame_*.png"))
bright = [(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).mean(), f) for f in frame_files]
ref = cv2.imread(str(max(bright, key=lambda x: x[0])[1]))

vis = ref.copy()
# Tum panelleri soluk yesille goster (baglam icin)
for pid, p in panels.items():
    pts = np.array(p["points"], np.int32)
    cv2.polylines(vis, [pts], True, (0, 80, 0), 1)

# Kume panellerini kalin sari cizgiyle vurgula
for pid in TARGET_CLUSTER_PANELS:
    pts = np.array(panels[pid]["points"], np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 255), 3)
    tl = pts[np.argmin(pts.sum(axis=1))]
    cv2.putText(vis, str(pid), (int(tl[0])+4, int(tl[1])+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

# Komsuluk kenarlarini kirmizi cizgiyle baglayarak ciz
for a, b, d, lim in edges_found:
    cv2.line(vis, all_centers[a], all_centers[b], (0, 0, 255), 2)

out = Path("6_labels_v2/cluster_inspect.png")
out.parent.mkdir(exist_ok=True)
cv2.imwrite(str(out), vis)
print(f"\nHarita: {out}")
print("Sari = kume paneli | Kirmizi cizgi = komsuluk baglantisi")