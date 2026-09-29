"""Net class change estimates and sensitivity of the main results to low-confidence reference labels."""
import json
import numpy as np, pandas as pd
from common import *
from estimators import Design, Z95

y1, y2 = YEARS
PIX_HA = CFG['project']['pixel_size_m'] ** 2 / 1e4
sp0 = pd.read_csv(f'{SAMP}/validation_points.csv').merge(pd.read_csv(f'{SAMP}/reference_labels.csv'), on='pid')
al = pd.read_csv(f'{TABS}/strata_allocation.csv', index_col=0)


def at(path, sp):
    return read(path)[0][0][sp.row.values, sp.col.values]


out = {}
for tag, sp in (('all', sp0), ('confidence>=2', sp0[sp0.confidence >= 2])):
    D = Design(sp.stratum.values, al.N_pixels.to_dict())
    r1, r2 = sp.ref_2018.values, sp.ref_2025.values
    m1 = at(f'{MAPS}/lulc_{y1}_rf.tif', sp); m2 = at(f'{MAPS}/lulc_{y2}_pcc.tif', sp)
    rc, mc = r1 != r2, m1 != m2
    rec = {'n': int(len(sp))}
    for k in CLASSES:
        p, se = D.proportion((r2 == k).astype(float) - (r1 == k).astype(float))
        rec[f'net_{CLASSES[k]}_ha'] = [p * D.N * PIX_HA, Z95 * se * D.N * PIX_HA]
    p, se = D.proportion(rc); rec['total_change_ha'] = [p * D.N * PIX_HA, Z95 * se * D.N * PIX_HA]
    rec['OA_2018'] = [D.proportion(m1 == r1)[0], Z95 * D.proportion(m1 == r1)[1]]
    rec['OA_2025'] = [D.proportion(m2 == r2)[0], Z95 * D.proportion(m2 == r2)[1]]
    ua = D.ratio(mc & rc, mc); pa = D.ratio(mc & rc, rc)
    rec['PCC_change_UA'] = [ua[0], Z95 * ua[1]]; rec['PCC_change_PA'] = [pa[0], Z95 * pa[1]]
    out[tag] = rec
json.dump(out, open(f'{TABS}/sensitivity_confidence.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
