#!/usr/bin/env python3
"""
PROJE KLASORUNU DUZENLEME SCRIPTI

Bu script, staj_proje klasorundeki dagitik dosya/klasorleri mantikli bir
hiyerarsiye tasir. Klasorler TEK PARCA tasindigi icin (icindeki binlerce
dosya tek tek degil) islem saniyeler surer.

GUVENLIK: Script once TASIMA PLANINI yazdirir, gercek islem yapmadan once
onay ister. Hicbir dosya SILINMEZ, sadece tasinir.

KULLANIM:
    python organize_project.py
"""
from pathlib import Path
import shutil

ROOT = Path(".").resolve()
print(f"Calisma klasoru: {ROOT}\n")

# ===== HEDEF YAPI: (kaynak, hedef) ciftleri =====
# Klasorler
FOLDER_MOVES = [
    ("1_Extract_frames",       "03_data/frames_video1"),
    ("1_Extract_frames_v2",    "03_data/frames_video2"),
    ("2_panel_data_manual",    "03_data/panel_template"),
    ("4_panel_crops",          "03_data/panel_crops_video1"),
    ("4_panel_crops_v2",       "03_data/panel_crops_video2"),

    ("5_hotspot",              "04_analysis/hotspot_video1"),
    ("5_hotspot_v2",           "04_analysis/hotspot_video2"),
    ("6_labels",               "04_analysis/labels_video1"),
    ("6_labels_v2",            "04_analysis/labels_video2"),
    ("7_dataset_split",        "04_analysis/dataset_split"),

    ("8_model",                "05_models/model_4class"),
    ("8_model_binary",         "05_models/model_binary"),

    ("9_end_to_end",           "06_results/end_to_end"),
    ("10_gradcam",             "06_results/gradcam"),

    ("2_debug",                "07_debug_outputs/debug_video1"),
    ("2_debug_v2",             "07_debug_outputs/debug_video2"),

    # Icerigi belirsiz / bos gorunen eski klasorler -> arsive
    ("syntax_deneme",          "arsiv/syntax_deneme"),
]

# Aktif adim scriptleri (1-10)
SCRIPT_MOVES = [
    ("1_extract_frames.py",        "01_scripts/1_extract_frames.py"),
    ("1_extract_frames_v2.py",     "01_scripts/1_extract_frames_v2.py"),
    ("3_manual_annotate.py",       "01_scripts/3_manual_annotate.py"),
    ("4_crop_panels.py",           "01_scripts/4_crop_panels.py"),
    ("4_crop_panels_v2.py",        "01_scripts/4_crop_panels_v2.py"),
    ("5_hotspot_profile.py",       "01_scripts/5_hotspot_profile.py"),
    ("5_hotspot_profile_v2.py",    "01_scripts/5_hotspot_profile_v2.py"),
    ("6_label_panels.py",          "01_scripts/6_label_panels.py"),
    ("6_label_panels_v2.py",       "01_scripts/6_label_panels_v2.py"),
    ("6c_fix_k1_labels.py",        "01_scripts/6c_fix_k1_labels.py"),
    ("7_split_dataset.py",         "01_scripts/7_split_dataset.py"),
    ("8_train_resnet18.py",        "01_scripts/8_train_resnet18.py"),
    ("8b_train_binary.py",         "01_scripts/8b_train_binary.py"),
    ("9_end_to_end_inference.py",  "01_scripts/9_end_to_end_inference.py"),
    ("10_gradcam.py",              "01_scripts/10_gradcam.py"),
]

# Gelistirme sirasinda kullanilan teshis/yardimci scriptler
DEBUG_SCRIPT_MOVES = [
    ("3c_check_template_v2.py",   "02_debug_tools/3c_check_template_v2.py"),
    ("5b_check_brightness.py",    "02_debug_tools/5b_check_brightness.py"),
    ("5b_debug_panel.py",         "02_debug_tools/5b_debug_panel.py"),
    ("5c_check_frame61.py",       "02_debug_tools/5c_check_frame61.py"),
    ("5d_diag_components.py",     "02_debug_tools/5d_diag_components.py"),
    ("6a_dilate_sweep.py",        "02_debug_tools/6a_dilate_sweep.py"),
    ("6a_ratio_sweep.py",         "02_debug_tools/6a_ratio_sweep.py"),
    ("6b_inspect_cluster.py",     "02_debug_tools/6b_inspect_cluster.py"),
    ("6b_trace_path.py",          "02_debug_tools/6b_trace_path.py"),
    ("inspect_k1.py",             "02_debug_tools/inspect_k1.py"),
]

# Veriler / CSV / videolar
DATA_MOVES = [
    ("video1.mp4",              "00_videos/video1.mp4"),
    ("video2.mp4",              "00_videos/video2.mp4"),
    ("4_panel_stats.csv",       "03_data/panel_stats_video1.csv"),
    ("4_panel_stats_v2.csv",    "03_data/panel_stats_video2.csv"),
]

ALL_MOVES = FOLDER_MOVES + SCRIPT_MOVES + DEBUG_SCRIPT_MOVES + DATA_MOVES

# Bos rapor klasoru (PDF'leri sen manuel koyacaksin)
EXTRA_DIRS = ["08_reports"]


# ===== 1) PLANI GOSTER =====
print("="*70)
print("TASIMA PLANI")
print("="*70)

existing_moves = []
missing = []

for src, dst in ALL_MOVES:
    src_path = ROOT / src
    if src_path.exists():
        existing_moves.append((src, dst))
        print(f"  {src:<32} -> {dst}")
    else:
        missing.append(src)

if missing:
    print(f"\nUYARI: Su kaynaklar bulunamadi, atlanacak:")
    for m in missing:
        print(f"    - {m}")

print(f"\nToplam tasinacak ogr: {len(existing_moves)}")
print(f"Ek olarak olusturulacak bos klasorler: {EXTRA_DIRS}")

# ===== 2) ONAY =====
answer = input("\nBu plani uygulamak istiyor musun? (evet/hayir): ").strip().lower()
if answer not in ("evet", "e", "yes", "y"):
    print("Iptal edildi. Hicbir dosya tasinmadi.")
    raise SystemExit(0)

# ===== 3) UYGULA =====
print("\n" + "="*70)
print("TASINIYOR...")
print("="*70)

for src, dst in existing_moves:
    src_path = ROOT / src
    dst_path = ROOT / dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        print(f"  ATLANDI (hedef zaten var): {dst}")
        continue

    shutil.move(str(src_path), str(dst_path))
    print(f"  OK: {src} -> {dst}")

for d in EXTRA_DIRS:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
    print(f"  OLUSTURULDU: {d}/")

print("\n" + "="*70)
print("TAMAMLANDI")
print("="*70)
print("\nNot: arsiv/ ve .vscode/ klasorlerine dokunulmadi.")
print("Not: 08_reports/ klasorunu, indirdigin PDF raporlarla")
print("     kendin doldurabilirsin.")