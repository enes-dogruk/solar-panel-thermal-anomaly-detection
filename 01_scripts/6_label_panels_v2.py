#!/usr/bin/env python3
"""
ADIM 6 (video2): KOMSULUK ANALIZI VE ETIKETLEME  (v2 - duzeltildi)
Video1 icin gelistirilen mantigin aynisi, video2 uzerinde.

SINIFLAR (Kursat bey terminolojisi):
  normal        : leke yok
  tekil_panel   : izole lekeli panel (kume boyutu 1)
  hat_arizasi   : bitisik, KOMPAKT lekeli panel grubu
  string_arizasi: bitisik, UZAMIS lekeli panel grubu (bir hat boyunca dizili)

DUZELTMELER (ilk denemeden sonra):
  1) KOMSULUK ESIGI: panelin KENDI capina gore degil (bu video1'de "hepsi
     tek kume" hatasina yol acmisti), TUM 143 panel arasindaki EN YAKIN
     KOMSU MESAFESINE (nn_dist) gore hesaplanir - video1'in dogrulanmis
     final yontemi.
  2) MIN_BLOB FILTRESI KALDIRILDI: nblob_med (120 karenin medyan leke
     sayisi) video2'de yanilticiydi, cunku arizalar ZAMANLA DEGISKEN
     (once saglam, sonra arizali) - panelin cogu karede saglam oldugu
     donemlerde medyan dusuk cikip gercek lekeli panelleri (33, 8, 17,
     49... gibi net gorunen lekeleri) yanlislikla eliyordu. Artik sadece
     HOT_THRESH kullanilir.
"""
import cv2, numpy as np, json, csv
from pathlib import Path

PROFILE   = Path("5_hotspot_v2/panel_profile.csv")
TEMPLATE  = Path("2_panel_data_manual/panels_manual.json")
FRAMES    = Path("1_Extract_frames_v2")
OUT_DIR   = Path("6_labels_v2")
OUT_DIR.mkdir(exist_ok=True)

# ---- AYARLAR ----
HOT_THRESH     = 0.02   # hot_p90 esigi
NEIGH_FACTOR   = 1.35   # komsuluk mesafesi = en_yakin_komsu_mesafesi * bu
ELONGATION_CUT = 3.0    # bu oranin uzeri "string", altı "hat_arizasi"

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}

rows = list(csv.DictReader(open(PROFILE)))
for r in rows:
    r["panel_id"]  = int(r["panel_id"])
    r["hot_p90"]   = float(r["hot_p90"])
    r["nblob_med"] = float(r["nblob_med"])
    r["cx"], r["cy"] = int(r["cx"]), int(r["cy"])

# ===== 1) LEKELI PANELLERI BELIRLE (sadece HOT_THRESH) =====
hot_ids = {r["panel_id"] for r in rows if r["hot_p90"] > HOT_THRESH}

print(f"Lekeli panel : {len(hot_ids)}")


# ===== 2) HER PANELIN "BIR GRID ADIMI" MESAFESINI OLC =====
# Panel capi DEGIL, TUM 143 panel arasindaki en yakin komsu mesafesi.
all_centers = {r["panel_id"]: (r["cx"], r["cy"]) for r in rows}

def nearest_neighbor_dist(pid):
    cx, cy = all_centers[pid]
    dists = [np.hypot(cx - ocx, cy - ocy)
             for opid, (ocx, ocy) in all_centers.items() if opid != pid]
    return min(dists) if dists else 999999

nn_dist = {r["panel_id"]: nearest_neighbor_dist(r["panel_id"]) for r in rows}
print(f"Ortalama en-yakin-komsu mesafesi: {np.mean(list(nn_dist.values())):.1f} px")


# ===== 3) KOMSULUK GRAFI (nn_dist tabanli) =====
def are_neighbors(a, b):
    ra = next(r for r in rows if r["panel_id"] == a)
    rb = next(r for r in rows if r["panel_id"] == b)
    d = np.hypot(ra["cx"] - rb["cx"], ra["cy"] - rb["cy"])
    lim = (nn_dist[a] + nn_dist[b]) / 2 * NEIGH_FACTOR
    return d < lim

hot_list = sorted(hot_ids)
adj = {i: set() for i in hot_list}
for i in range(len(hot_list)):
    for j in range(i + 1, len(hot_list)):
        a, b = hot_list[i], hot_list[j]
        if are_neighbors(a, b):
            adj[a].add(b)
            adj[b].add(a)


# ===== 4) BAGLANTILI BILESENLER =====
seen, clusters = set(), []
for start in hot_list:
    if start in seen:
        continue
    stack, comp = [start], []
    seen.add(start)
    while stack:
        n = stack.pop()
        comp.append(n)
        for m in adj[n]:
            if m not in seen:
                seen.add(m)
                stack.append(m)
    clusters.append(sorted(comp))

clusters.sort(key=len, reverse=True)


# ===== 5) UZAMA ORANI HESAPLA (PCA) =====
def elongation_ratio(pids):
    if len(pids) < 3:
        centers = np.array([[r["cx"], r["cy"]] for r in rows if r["panel_id"] in pids])
        dist = np.hypot(*(centers[0] - centers[1]))
        avg_nn = np.mean([nn_dist[p] for p in pids])
        return dist / avg_nn if avg_nn > 0 else 1.0

    centers = np.array([[r["cx"], r["cy"]] for r in rows if r["panel_id"] in pids],
                       dtype=np.float64)
    centers -= centers.mean(axis=0)
    cov = np.cov(centers.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return float(eigvals[-1] / eigvals[0])


cluster_shape = {}
for ci, comp in enumerate(clusters):
    if len(comp) >= 2:
        cluster_shape[ci] = elongation_ratio(comp)


# ===== 6) ETIKETLE =====
label_of, cluster_of = {}, {}
for ci, comp in enumerate(clusters):
    if len(comp) == 1:
        lab = "tekil_panel"
    else:
        ratio = cluster_shape[ci]
        lab = "string_arizasi" if ratio > ELONGATION_CUT else "hat_arizasi"
    for pid in comp:
        label_of[pid] = lab
        cluster_of[pid] = ci

for r in rows:
    pid = r["panel_id"]
    r["label"] = label_of.get(pid, "normal")
    r["cluster_id"] = cluster_of.get(pid, -1)
    r["cluster_size"] = len(clusters[cluster_of[pid]]) if pid in cluster_of else 0

out_csv = OUT_DIR / "panel_labels.csv"
fields = ["panel_id", "cx", "cy", "hot_p90", "hot_med", "hot_max",
          "hot_stable", "nblob_med", "big_med", "spread_med",
          "label", "cluster_id", "cluster_size"]
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["panel_id"]))

json.dump({str(r["panel_id"]): r["label"] for r in rows},
          open(OUT_DIR / "labels.json", "w"), indent=2)


# ===== 7) DOGRULAMA HARITASI =====
COLOR_NORMAL       = (0, 200, 0)
COLOR_TEKIL        = (255, 255, 255)
COLOR_HAT_ARIZASI  = (0, 140, 255)
COLOR_STRING       = (0, 0, 255)

frame_files = sorted(FRAMES.glob("frame_*.png"))
bright = [(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).mean(), f) for f in frame_files]
ref = cv2.imread(str(max(bright, key=lambda x: x[0])[1]))

vis = ref.copy()
for r in rows:
    pid = r["panel_id"]
    pts = np.array(panels[pid]["points"], np.int32)

    if r["label"] == "normal":
        col, th = COLOR_NORMAL, 1
    elif r["label"] == "tekil_panel":
        col, th = COLOR_TEKIL, 3
    elif r["label"] == "hat_arizasi":
        col, th = COLOR_HAT_ARIZASI, 3
    else:
        col, th = COLOR_STRING, 3

    cv2.polylines(vis, [pts], True, col, th)
    tl = pts[np.argmin(pts.sum(axis=1))]
    cv2.putText(vis, str(pid), (int(tl[0]) + 4, int(tl[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    if r["cluster_id"] != -1 and r["cluster_size"] >= 2:
        same_cluster = [q for q in rows if q["cluster_id"] == r["cluster_id"]]
        if r["cx"] == min(q["cx"] for q in same_cluster):
            label_col = COLOR_STRING if r["label"] == "string_arizasi" else COLOR_HAT_ARIZASI
            cv2.putText(vis, f"K{r['cluster_id']+1}", (int(tl[0]) + 4, int(tl[1]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_col, 1)

y = 26
cv2.rectangle(vis, (8, 8), (360, 132), (0, 0, 0), -1)
for txt, col in [("normal", COLOR_NORMAL),
                 ("tekil_panel", COLOR_TEKIL),
                 ("hat_arizasi (kompakt kume)", COLOR_HAT_ARIZASI),
                 ("string_arizasi (uzamis hat)", COLOR_STRING)]:
    cv2.putText(vis, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    y += 26
cv2.imwrite(str(OUT_DIR / "label_map.png"), vis)


# ===== OZET =====
from collections import Counter
cnt = Counter(r["label"] for r in rows)

print("\n" + "="*60)
print("  SINIF DAGILIMI")
print("="*60)
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    n = cnt.get(lab, 0)
    print(f"    {lab:<16}: {n:>3} panel  ({n/len(rows)*100:>5.1f}%)")
print(f"    {'TOPLAM':<16}: {len(rows):>3}")

print(f"\n  Kumeler ve tipleri:")
for ci, comp in enumerate(clusters):
    if len(comp) >= 2:
        ratio = cluster_shape[ci]
        lab = "string_arizasi" if ratio > ELONGATION_CUT else "hat_arizasi"
        print(f"    K{ci+1}: {len(comp)} panel {comp} "
              f"-> {lab} (uzama orani: {ratio:.2f})")

izole = [c[0] for c in clusters if len(c) == 1]
print(f"\n  Izole (tekil_panel) sayisi: {len(izole)}")
print(f"    {izole}")

print(f"\n  Egitim ornegi (120 kare x panel):")
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    print(f"    {lab:<16}: {cnt.get(lab,0) * 120:>6}")

print(f"\n  CSV    : {out_csv}")
print(f"  JSON   : {OUT_DIR}/labels.json")
print(f"  Harita : {OUT_DIR}/label_map.png  <- GOZLE DOGRULA")
print("="*60)