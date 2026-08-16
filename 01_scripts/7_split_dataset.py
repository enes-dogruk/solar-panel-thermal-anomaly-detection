#!/usr/bin/env python3
"""
ADIM 7: TRAIN / VAL / TEST BOLMESI  (video1 + video2 birlesik)  -- v2

BOLUNMEZ BIRIM = PANEL NUMARASI (video degil!)
  Ayni panel numarasinin iki videodaki hali FIZIKSEL OLARAK AYNI PANELDIR.
  Biri train'de digeri test'te olursa model panelin KIMLIGINI ezberleyip
  ariza tipini degil paneli taniyabilir -> veri sizintisi. Panel 8'in hem
  video1 hem video2 versiyonu HEP AYNI SETE gider.

EK KISIT: hat_arizasi / string_arizasi KUMELERI de bolunmez.

v2 DUZELTMESI: Ilk denemede greedy algoritma buyuk birimleri once
  yerlestirdigi icin string_arizasi (sadece birkac buyuk birimden olusan
  kucuk bir sinif) TEST SETINE HIC BIRIM DUSMEDEN tukendi (train=14, val=3,
  test=0). Cozum: her sinif icin ONCE her sete EN AZ 1 BIRIM garanti
  edilir (yeterli birim varsa), SONRA kalan birimler normal greedy
  dengeleme ile dagitilir.
"""
import csv, json, random
from pathlib import Path
from collections import defaultdict, Counter

LABELS_V1 = Path("6_labels/panel_labels.csv")
LABELS_V2 = Path("6_labels_v2/panel_labels.csv")
CROPS_V1  = Path("4_panel_crops")
CROPS_V2  = Path("4_panel_crops_v2")
OUT_DIR   = Path("7_dataset_split")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SEED = 42
random.seed(SEED)

# ===== 1) IKI VIDEONUN ETIKETLERINI OKU =====
def load_labels(path, video_tag):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        out.append({
            "video": video_tag,
            "panel_id": int(r["panel_id"]),
            "label": r["label"],
            "cluster_id": int(r["cluster_id"]),
            "sample_id": f"{video_tag}_p{int(r['panel_id']):03d}",
        })
    return out

recs_v1 = load_labels(LABELS_V1, "video1")
recs_v2 = load_labels(LABELS_V2, "video2")
all_recs = recs_v1 + recs_v2

print(f"video1: {len(recs_v1)} panel | video2: {len(recs_v2)} panel")
print(f"Toplam ornek (panel-video cifti): {len(all_recs)}\n")

print("SINIF DAGILIMI (birlesik):")
print(f"{'sinif':<16}{'video1':>8}{'video2':>8}{'TOPLAM':>8}")
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    n1 = sum(1 for r in recs_v1 if r["label"] == lab)
    n2 = sum(1 for r in recs_v2 if r["label"] == lab)
    print(f"{lab:<16}{n1:>8}{n2:>8}{n1+n2:>8}")


# ===== 2) BOLUNMEZ BIRIMLERI OLUSTUR =====
panel_to_recs = defaultdict(list)
for r in all_recs:
    panel_to_recs[r["panel_id"]].append(r)

parent = {pid: pid for pid in panel_to_recs}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for recs in (recs_v1, recs_v2):
    clusters = defaultdict(list)
    for r in recs:
        if r["cluster_id"] != -1:
            clusters[r["cluster_id"]].append(r["panel_id"])
    for pids in clusters.values():
        for p in pids[1:]:
            union(pids[0], p)

groups = defaultdict(list)
for pid in panel_to_recs:
    groups[find(pid)].append(pid)

units = []
for root, pids in groups.items():
    recs = [r for p in pids for r in panel_to_recs[p]]
    labs = [r["label"] for r in recs if r["label"] != "normal"]
    dominant = Counter(labs).most_common(1)[0][0] if labs else "normal"
    units.append({"panels": sorted(pids), "recs": recs,
                  "size": len(recs), "dominant": dominant})

print(f"\nBolunmez birim sayisi: {len(units)}")

print(f"\nSinif basina birim sayisi (dagitim zorlugu icin onemli):")
unit_counts = Counter(u["dominant"] for u in units)
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    print(f"    {lab:<16}: {unit_counts.get(lab, 0)} birim")


# ===== 3) SINIF BAZINDA DENGELI DAGIT (v2: garanti + greedy) =====
targets = {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC}
split_of_panel = {}

by_class = defaultdict(list)
for u in units:
    by_class[u["dominant"]].append(u)

print(f"\nDagitim raporu:")
for lab, us in by_class.items():
    us_sorted = sorted(us, key=lambda u: -u["size"])
    random.shuffle(us_sorted)
    us_sorted.sort(key=lambda u: -u["size"])

    total = sum(u["size"] for u in us_sorted)
    target_counts = {s: total * f for s, f in targets.items()}
    assigned = {s: 0 for s in targets}
    assigned_units = {s: 0 for s in targets}

    remaining = list(us_sorted)

    # --- ONCE: her sete en az 1 birim garanti et (yeterli birim varsa) ---
    # SIRA ONEMLI: train en buyuk payi hak ettigi icin ONCE train garantilenir
    # (en buyuk kalan birim train'e gider), sonra val, sonra test. Ilk
    # denemede sira ters (test->val->train) oldugu icin string_arizasi gibi
    # az birimli siniflarda en buyuk birim yanlislikla test'e, en kucugu
    # train'e gidiyordu (train < val gibi mantiksiz bir sonuc olusuyordu).
    guarantee_order = ["train", "val", "test"]
    if len(remaining) >= 3:
        for s in guarantee_order:
            if remaining:
                u = remaining.pop(0)  # en buyuk kalan birimi ver
                for pid in u["panels"]:
                    split_of_panel[pid] = s
                assigned[s] += u["size"]
                assigned_units[s] += 1
    elif len(remaining) > 0:
        # cok az birim var (1-2), guvenli tarafta train'e ver
        print(f"    UYARI: '{lab}' sadece {len(remaining)} birim icerdigi icin "
              f"her sete garanti dagitilamiyor.")

    # --- SONRA: kalan birimleri greedy dengele ---
    for u in remaining:
        deficits = {s: target_counts[s] - assigned[s] for s in targets}
        best = max(deficits, key=deficits.get)
        assigned[best] += u["size"]
        assigned_units[best] += 1
        for pid in u["panels"]:
            split_of_panel[pid] = best

    print(f"    {lab:<16}: birim -> train={assigned_units['train']} "
          f"val={assigned_units['val']} test={assigned_units['test']} "
          f"| panel -> train={assigned['train']} val={assigned['val']} "
          f"test={assigned['test']}")


# ===== 4) CIKTI: CSV + JSON =====
out_csv = OUT_DIR / "sample_split.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["sample_id", "video", "panel_id",
                                        "label", "cluster_id", "split"])
    w.writeheader()
    for r in sorted(all_recs, key=lambda r: (r["video"], r["panel_id"])):
        w.writerow({**{k: r[k] for k in ["sample_id", "video", "panel_id",
                                          "label", "cluster_id"]},
                    "split": split_of_panel[r["panel_id"]]})

json.dump({str(k): v for k, v in split_of_panel.items()},
          open(OUT_DIR / "panel_split.json", "w"), indent=2)


# ===== 5) KIRPIM DOSYA LISTELERI =====
handles = {s: open(OUT_DIR / f"{s}.txt", "w") for s in ["train", "val", "test"]}
list_counts = Counter()

label_lookup = {(r["video"], r["panel_id"]): r["label"] for r in all_recs}

for crops_dir, tag in [(CROPS_V1, "video1"), (CROPS_V2, "video2")]:
    if not crops_dir.exists():
        print(f"\nUYARI: {crops_dir}/ bulunamadi, atlaniyor.")
        continue
    for f in sorted(crops_dir.glob("*.png")):
        try:
            pid = int(f.stem.split("_p")[1])
        except (IndexError, ValueError):
            continue
        if pid not in split_of_panel:
            continue
        lab = label_lookup.get((tag, pid))
        if lab is None:
            continue
        s = split_of_panel[pid]
        handles[s].write(f"{crops_dir.name}/{f.name}\t{lab}\n")
        list_counts[(s, lab)] += 1
        list_counts[s] += 1

for h in handles.values():
    h.close()


# ===== OZET =====
print("\n" + "="*62)
print("  PANEL-VIDEO ORNEGI BAZINDA DAGILIM")
print("="*62)
print(f"{'sinif':<16}{'train':>9}{'val':>7}{'test':>7}{'TOPLAM':>9}")
print("-"*62)
for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
    c = Counter(split_of_panel[r["panel_id"]] for r in all_recs if r["label"] == lab)
    tot = sum(c.values())
    print(f"{lab:<16}{c.get('train',0):>9}{c.get('val',0):>7}"
          f"{c.get('test',0):>7}{tot:>9}")

tot_c = Counter(split_of_panel[r["panel_id"]] for r in all_recs)
print("-"*62)
print(f"{'TOPLAM':<16}{tot_c.get('train',0):>9}{tot_c.get('val',0):>7}"
      f"{tot_c.get('test',0):>7}{len(all_recs):>9}")

if list_counts:
    print(f"\n  GORUNTU (kirpim) SAYISI:")
    print(f"{'sinif':<16}{'train':>9}{'val':>7}{'test':>7}")
    print("-"*40)
    for lab in ["normal", "tekil_panel", "hat_arizasi", "string_arizasi"]:
        print(f"{lab:<16}{list_counts[('train',lab)]:>9}"
              f"{list_counts[('val',lab)]:>7}{list_counts[('test',lab)]:>7}")
    print("-"*40)
    print(f"{'TOPLAM':<16}{list_counts['train']:>9}"
          f"{list_counts['val']:>7}{list_counts['test']:>7}")
    print(f"\n  GENEL TOPLAM: "
          f"{list_counts['train']+list_counts['val']+list_counts['test']} goruntu")

print(f"\n  CSV   : {out_csv}")
print(f"  JSON  : {OUT_DIR}/panel_split.json")
print(f"  Liste : {OUT_DIR}/train.txt, val.txt, test.txt")
print("="*62)