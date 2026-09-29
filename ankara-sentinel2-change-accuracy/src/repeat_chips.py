"""Sheets for a blind repeat interpretation of a random 20% subset (new IDs, shuffled order)."""
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
import reference_chips as R
from common import *

sp = pd.read_csv(f'{SAMP}/validation_points.csv')
rng = np.random.default_rng(2024)
sub = sp.iloc[rng.choice(len(sp), 100, replace=False)].copy()
sub['rid'] = np.arange(1, 101)
sub[['rid', 'pid']].to_csv(f'{SAMP}/repeat_key.csv', index=False)
out = os.path.join(ROOT, 'data', 'reference', 'repeat_sheets'); os.makedirs(out, exist_ok=True)
feats = {y: read(f'{PROC}/S2_{y}_features.tif') for y in YEARS}
font = ImageFont.truetype('arial.ttf', 14)
CH = R.CH
for s in range(0, 100, 6):
    part = sub.iloc[s:s + 6]
    sheet = Image.new('RGB', (40 + 6 * (CH + 4), 22 + len(part) * (CH + 4)), 'white'); d = ImageDraw.Draw(sheet)
    for j, t in enumerate(['VHR 2018', 'S2 2018 summer', 'S2 2018 spring FCC', 'VHR 2025', 'S2 2025 summer', 'S2 2025 spring FCC']):
        d.text((40 + j * (CH + 4) + 4, 3), t, fill='black', font=font)
    for i, (_, p) in enumerate(part.iterrows()):
        yy = 22 + i * (CH + 4); d.text((2, yy + CH // 2 - 7), f'R{p.rid}', fill='black', font=font)
        chips = []
        for k, yr in (('T1', YEARS[0]), ('T2', YEARS[1])):
            arr, _, names = feats[yr]
            chips += [R.vhr_chip(R.REL[k], p.x, p.y),
                      R.s2_chip(arr, names, p.row, p.col, ['B4_summer', 'B3_summer', 'B2_summer'], 3.2),
                      R.s2_chip(arr, names, p.row, p.col, ['B8_spring', 'B4_spring', 'B3_spring'], 2.4)]
        for j, ch in enumerate(chips):
            sheet.paste(ch, (40 + j * (CH + 4), yy))
    sheet.save(os.path.join(out, f'repeat_{s // 6 + 1:02d}.jpg'), quality=88)
print('ok')
