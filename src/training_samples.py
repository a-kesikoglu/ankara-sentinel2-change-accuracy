"""Automated training sample extraction from temporally stable, multi-source consensus pixels.

A pixel is eligible when Dynamic World (seasonal mode, 2018 and 2025) and ESA WorldCover 2021
assign the same study class, the Dynamic World probability of that class is at least the
configured threshold in both years, and the 3x3 neighbourhood is homogeneous in both years.
Eligible pixels are thinned to one candidate per 10 x 10 pixel block (100 m) and sampled per class.
"""
import numpy as np, pandas as pd
from scipy import ndimage
from common import *

rng = np.random.default_rng(SEED)
y1, y2 = YEARS
dw1 = DW_MAP[read(f'{PROC}/DW_{y1}_label.tif')[0][0]]
dw2 = DW_MAP[read(f'{PROC}/DW_{y2}_label.tif')[0][0]]
wc = remap_wc(read(f'{PROC}/WC_2021.tif')[0][0])
p1 = read(f'{PROC}/DW_{y1}_prob.tif')[0].astype(np.float32) / 100
p2 = read(f'{PROC}/DW_{y2}_prob.tif')[0].astype(np.float32) / 100


def class_prob(p):
    # order: water, trees, grass, flooded_vegetation, crops, shrub_and_scrub, built, bare, snow
    return np.stack([p[0], p[1], p[2] + p[3] + p[4] + p[5], p[6], p[7]])


cp1, cp2 = class_prob(p1), class_prob(p2)
thr = CFG['training']['dw_min_probability']


def homogeneous(lab):
    mx = ndimage.maximum_filter(lab, 3)
    mn = ndimage.minimum_filter(lab, 3)
    return mx == mn


ok = (dw1 == dw2) & (dw1 == wc) & (dw1 > 0) & homogeneous(dw1) & homogeneous(dw2)
# GHSL 2018 built-up surface (m2 per 10 m cell): built-up samples must contain buildings in the
# 3x3 neighbourhood, all other classes must be free of building surface in it
ghs = ndimage.maximum_filter(read(f'{PROC}/GHSL_BUILT_S_2018.tif')[0][0], 3)
ok &= np.where(dw1 == 4, ghs >= 20, ghs == 0)
H, W = ok.shape
rows, cols = np.nonzero(ok)
lab = dw1[rows, cols]
pr1 = cp1[lab - 1, rows, cols]
pr2 = cp2[lab - 1, rows, cols]
keep = (pr1 >= thr) & (pr2 >= thr)
rows, cols, lab = rows[keep], cols[keep], lab[keep]

# one random candidate per 100 m block (30 m for the spatially limited water class)
bs = np.where(lab == 1, 3, 10)
block = lab.astype(np.int64) * 10**9 + (rows // bs) * (W + 1) + cols // bs
order = rng.permutation(len(rows))
_, first = np.unique(block[order], return_index=True)
sel = order[first]
rows, cols, lab = rows[sel], cols[sel], lab[sel]

n = CFG['training']['samples_per_class_per_year']
recs = []
for c in CLASSES:
    idx = np.nonzero(lab == c)[0]
    take = rng.choice(idx, size=min(n, len(idx)), replace=False)
    recs.append(pd.DataFrame({'row': rows[take], 'col': cols[take], 'class_id': c}))
df = pd.concat(recs, ignore_index=True)

_, prof, _ = read(f'{PROC}/WC_2021.tif')
T = prof['transform']
df['x'] = T.c + (df.col + 0.5) * T.a
df['y'] = T.f + (df.row + 0.5) * T.e
df['class_name'] = df.class_id.map(CLASSES)
df.to_csv(f'{SAMP}/training_points.csv', index=False)
summary = pd.DataFrame({'eligible_pixels': [int((ok & (dw1 == c)).sum()) for c in CLASSES],
                        'training_points': [int((df.class_id == c).sum()) for c in CLASSES]},
                       index=[CLASSES[c] for c in CLASSES])
summary.to_csv(f'{TABS}/training_sample_counts.csv')
print(summary)
