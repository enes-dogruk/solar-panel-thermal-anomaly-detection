#!/usr/bin/env python3
"""
ADIM 6: KOMSULUK ANALIZI VE ETIKETLEME  (v5 - DOGRU FINAL)
Lekeli panelleri komsuluk grafina gore siniflandirir.

SINIFLAR (Kursat bey terminolojisi):
  normal        : leke yok
  tekil_panel   : izole lekeli panel
  hat_arizasi   : bitisik 2+ lekeli panel

NOT: string_arizasi sinifi bu veride olusmadi (bkz. rapor notu).

GECMIS (v1->v5):
  v1: panel_capi * 1.6            -> hepsi tek kumede birlesti (esik cok gevsek)
  v2: nn_dist * 1.35               -> gorsel olarak saglikli (6 kume, 17 tekil) [DOGRU]
  v3: geometrik temas (dilate)     -> perspektif yuzunden asiri agresif, 14'lu
                                       dev kume olustu (sabit px, olcek degildi)
  v4: (HATA) yanlislikla v1'e donuldu (panel_capi), 33'lu tek kume tekrar cikti.
  v5: v2'ye DOGRU sekilde geri donuldu. nn_dist = 143 panelin TAMAMI arasindaki
      en yakin komsu mesafesi (panelin kendi capi degil). BFS ile 56-100
      baglantisi ayrica dogrulandi: dogrudan komsu cikti, hata degildi.
"""
import cv2, numpy as np, json, csv
from pathlib import Path

PROFILE   = Path("5_hotspot/panel_profile.csv")
TEMPLATE  = Path("2_panel_data_manual/panels_manual.json")
FRAMES    = Path("1_Extract_frames")
OUT_DIR   = Path("6_labels")
OUT_DIR.mkdir(exist_ok=True)

# ---- AYARLAR ----
HOT_THRESH   = 0.01    # hot_p90 esigi
MIN_BLOB     = 1       # en az 1 gercek leke sart (38, 93 gibi sizintilari eler)
NEIGH_FACTOR = 1.35    # komsuluk mesafesi = en_yakin_komsu_mesafesi * bu

panels = {p["panel_id"]: p for p in json.load(open(TEMPLATE))}

rows = list(csv.DictReader(open(PROFILE)))
for r in rows:
    r["panel_id"]  = int(r["panel_id"])
    r["hot_p90"]   = float(r["hot_p90"])
    r["nblob_med"] = float(r["nblob_med"])
    r["cx"], r["cy"] = int(r["cx"]), int(r["cy"])

# ===== 1) LEKELI PANELLERI BELIRLE =====
hot_ids = {r["panel_id"] for r in rows
           if r["hot_p90"] > HOT_THRESH and r["nblob_med"] >= MIN_BLOB}

elenen = [r["panel_id"] for r in rows
          if r["hot_p90"] > HOT_THRESH and r["nblob_med"] < MIN_BLOB]

print(f"Lekeli panel : {len(hot_ids)}")
if elenen:
    print(f"Blob filtresi ile elenen: {elenen}")


# ===== 2) HER PANELIN "BIR GRID ADIMI" MESAFESINI OLC =====
# Panel capi DEGIL, TUM 143 panel arasindaki en yakin komsu mesafesi kullanilir.
# Bu, perspektife otomatik uyum saglar (panel capi kullanmak asiri gevsek
# esige yol aciyordu -> tum lekeli paneller tek kumede birlesiyordu).
all_centers = {r["panel_id"]: (r["cx"], r["cy"]) for r in rows}

def nearest_neighbor_dist(pid):
    cx, cy = all_centers[pid]
    dists = [np.hypot(cx - ocx, cy - ocy)
             for opid, (ocx, ocy) in all_centers.items() if opid != pid]
    return min(dists) if dists else 999999

nn_dist = {r["panel_id"]: nearest_neighbor_dist(r["panel_id"]) for r in rows}

print(f"Ortalama en-yakin-komsu mesafesi: {np.mean(list(nn_dist.values())):.1f} px")
print(f"Min/Max: {min(nn_dist.values()):.1f} / {max(nn_dist.values()):.1f} px")


# ===== 3) KOMSULUK GRAFI (nn_dist tabanli, sadece lekeli paneller arasinda) =====
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


# ===== 5) ETIKETLE =====
label_of, cluster_of = {}, {}
for ci, comp in enumerate(clusters):
    lab = "hat_arizasi" if len(comp) >= 2 else "tekil_panel"
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


# ===== 6) DOGRULAMA HARITASI =====
# Renkler TIP bazlidir (kume bazli degil): ayni sinif = ayni renk.
# Boylece "hat_arizasi" kumeleri farkli renklerde gorunup kafa karistirmiyor;
# her renk tek bir anlam tasiyor.
COLOR_NORMAL      = (0, 200, 0)     # yesil
COLOR_TEKIL       = (255, 255, 255) # beyaz
COLOR_HAT_ARIZASI = (0, 140, 255)   # turuncu (BGR)

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
    else:
        col, th = COLOR_HAT_ARIZASI, 3

    cv2.polylines(vis, [pts], True, col, th)
    tl = pts[np.argmin(pts.sum(axis=1))]
    cv2.putText(vis, str(pid), (int(tl[0]) + 4, int(tl[1]) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # Hat_arizasi kumelerinde, kumenin en ust-sol panelinin yanina kume
    # numarasini kucuk bir etiketle yaz (K1, K2, ...) - hangi panellerin
    # ayni gruba ait oldugunu ayirt etmek icin, renk karmasasi olmadan.
    if r["label"] == "hat_arizasi" and r["cx"] == min(
            (q["cx"] for q in rows if q.get("cluster_id") == r["cluster_id"]),
            default=r["cx"]):
        cv2.putText(vis, f"K{r['cluster_id']+1}", (int(tl[0]) + 4, int(tl[1]) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_HAT_ARIZASI, 1)

y = 26
cv2.rectangle(vis, (8, 8), (360, 108), (0, 0, 0), -1)
for txt, col in [("normal", COLOR_NORMAL),
                 ("tekil_panel", COLOR_TEKIL),
                 ("hat_arizasi (Kn = grup no)", COLOR_HAT_ARIZASI)]:
    cv2.putText(vis, txt, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    y += 28
cv2.imwrite(str(OUT_DIR / "label_map.png"), vis)


# ===== OZET =====
from collections import Counter
cnt = Counter(r["label"] for r in rows)

print("\n" + "="*60)
print("  SINIF DAGILIMI")
print("="*60)
for lab in ["normal", "tekil_panel", "hat_arizasi"]:
    n = cnt.get(lab, 0)
    print(f"    {lab:<14}: {n:>3} panel  ({n/len(rows)*100:>5.1f}%)")
print(f"    {'TOPLAM':<14}: {len(rows):>3}")

print(f"\n  Kumeler ({len([c for c in clusters if len(c) >= 2])} adet):")
for c in clusters:
    if len(c) >= 2:
        print(f"    {len(c)} panel: {c}")

izole = [c[0] for c in clusters if len(c) == 1]
print(f"\n  Izole paneller ({len(izole)}): {izole}")

print(f"\n  Egitim ornegi (121 kare x panel):")
for lab in ["normal", "tekil_panel", "hat_arizasi"]:
    print(f"    {lab:<14}: {cnt.get(lab,0) * 121:>6}")

print(f"\n  CSV    : {out_csv}")
print(f"  JSON   : {OUT_DIR}/labels.json")
print(f"  Harita : {OUT_DIR}/label_map.png")
print("="*60)