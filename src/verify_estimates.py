"""Recompute the main accuracy and area estimates of the article from the files in this repository only
(maps in outputs/maps, samples in data/samples); no Earth Engine access is needed."""
import numpy as np, pandas as pd
from common import *
from estimators import Design, accuracy, Z95

PIX_HA = CFG['project']['pixel_size_m'] ** 2 / 1e4
sp = pd.read_csv(f'{SAMP}/validation_points.csv').merge(pd.read_csv(f'{SAMP}/reference_labels.csv'), on='pid')
al = pd.read_csv(f'{TABS}/strata_allocation.csv', index_col=0)
D = Design(sp.stratum.values, al.N_pixels.to_dict())
at = lambda f: read(f'{MAPS}/{f}')[0][0][sp.row.values, sp.col.values]
m18, m25 = at('lulc_2018_rf.tif'), at('lulc_2025_pcc.tif')
r18, r25 = sp.ref_2018.values, sp.ref_2025.values
for name, m, r in (('RF 2018', m18, r18), ('RF 2025 (PCC)', m25, r25)):
    (oa, se), _, _ = accuracy(D, m, r, list(CLASSES))
    print(f'{name}: overall accuracy {100 * oa:.1f} ± {100 * Z95 * se:.1f}%')
mc, rc = m18 != m25, r18 != r25
ua = D.ratio(mc & rc, mc); pa = D.ratio(mc & rc, rc)
print(f'PCC change: UA {100 * ua[0]:.1f} ± {100 * Z95 * ua[1]:.1f}%, PA {100 * pa[0]:.1f} ± {100 * Z95 * pa[1]:.1f}%')
p, se = D.proportion(rc)
print(f'Total change: {p * D.N * PIX_HA:,.0f} ± {Z95 * se * D.N * PIX_HA:,.0f} ha')
p, se = D.proportion((r25 == 4).astype(float) - (r18 == 4))
print(f'Net built-up change: {p * D.N * PIX_HA:,.0f} ± {Z95 * se * D.N * PIX_HA:,.0f} ha')
