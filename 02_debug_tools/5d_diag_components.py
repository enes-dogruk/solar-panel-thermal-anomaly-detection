#!/usr/bin/env python3
"""
TANI 3: hot_core ve hot_ring bilesenlerini AYRI AYRI kaydeder.
Boylece VERY_BRIGHT gercekten calisiyorsa hot_core.png'de leke govdelerinin
DOLU disk olarak gorunmesi gerekir. Eger hala simit ise, sorun VERY_BRIGHT
degil, kodun kaydedilmemis/calistirilmamis olmasi ihtimali yuksektir.
"""
import cv2, numpy as np
from pathlib import Path

FRAME = Path("1_Extract_frames_v2/frame_00061.png")
OUT   = Path("5_hotspot_v2")
OUT.mkdir(exist_ok=True)

# BURAYA 5_hotspot_profile_v2.py'deki GUNCEL degerlerini AYNEN yapistir:
WARM_HUE_LO  = 0
WARM_HUE_HI  = 35
WARM_WRAP_LO = 165
SAT_MIN      = 55
VAL_MIN      = 70
VERY_BRIGHT  = 150     # <-- bunun gercekten 150 oldugundan emin ol

img = cv2.imread(str(FRAME))
if img is None:
    print(f"HATA: {FRAME} bulunamadi.")
    raise SystemExit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

warm_hue = ((h >= WARM_HUE_LO) & (h <= WARM_HUE_HI)) | (h >= WARM_WRAP_LO)
hot_ring = warm_hue & (s >= SAT_MIN) & (v >= VAL_MIN)
hot_core = (v >= VERY_BRIGHT)
hot_both = hot_core | hot_ring

cv2.imwrite(str(OUT / "diag_hot_core_only.png"), np.uint8(hot_core) * 255)
cv2.imwrite(str(OUT / "diag_hot_ring_only.png"), np.uint8(hot_ring) * 255)
cv2.imwrite(str(OUT / "diag_hot_combined.png"), np.uint8(hot_both) * 255)

print(f"VERY_BRIGHT = {VERY_BRIGHT}  (bu deger dogru mu kontrol et!)")
print(f"\nhot_core piksel sayisi (V >= {VERY_BRIGHT}): {int(hot_core.sum())}")
print(f"hot_ring piksel sayisi (hue+doygunluk)  : {int(hot_ring.sum())}")
print(f"birlesim (OR)                            : {int(hot_both.sum())}")
print(f"\nKaydedildi:")
print(f"  {OUT}/diag_hot_core_only.png   <- SADECE parlaklik esigi")
print(f"  {OUT}/diag_hot_ring_only.png   <- SADECE eski hue+doygunluk")
print(f"  {OUT}/diag_hot_combined.png    <- ikisinin birlesimi")