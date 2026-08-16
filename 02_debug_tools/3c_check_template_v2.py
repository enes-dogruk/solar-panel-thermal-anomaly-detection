#!/usr/bin/env python3
"""
Mevcut panel sablonunu (video1'den) video2'nin ilk karesi uzerine
cizip gorsel uyum kontrolu yapar. Kaymis panelleri elle isaretlemek
gerekip gerekmedigine buradan karar verilir.
"""
import cv2, numpy as np, json
from pathlib import Path

TEMPLATE = Path("2_panel_data_manual/panels_manual.json")
FRAME_V2 = Path("1_Extract_frames_v2/frame_00000.png")
OUT_DIR  = Path("2_debug_v2")
OUT_DIR.mkdir(exist_ok=True)

panels = json.load(open(TEMPLATE))
img = cv2.imread(str(FRAME_V2))

if img is None:
    print(f"HATA: {FRAME_V2} bulunamadi. Once 1_extract_frames_v2.py calistir.")
    raise SystemExit(1)

vis = img.copy()
for p in panels:
    pts = np.array(p["points"], np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    c = pts.mean(axis=0).astype(int)
    cv2.putText(vis, str(p["panel_id"]), tuple(c),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

out_path = OUT_DIR / "template_on_video2.png"
cv2.imwrite(str(out_path), vis)
print(f"Kaydedildi: {out_path}")
print(f"Toplam panel: {len(panels)}")
print("\nBu goruntuyu ac ve kontrol et:")
print("  - Yesil cizgiler panellerin GERCEK kenarlarina mi oturuyor?")
print("  - Kayma varsa hangi bolgede (ust/alt/sol/sag)?")