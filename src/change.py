"""Post-classification comparison (PCC), PCC constrained by a change vector analysis (CVA)
mask with an Otsu threshold (CPCC), minimum mapping unit, transition matrices and the
stratified random validation sample design."""
import json
import numpy as np, pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu
from pyproj import Transformer
from common import *

MMU = CFG['postprocessing']['mmu_pixels']
PIX_HA = CFG['project']['pixel_size_m'] ** 2 / 1e4
y1, y2 = YEARS
CVA_BANDS = [f'{b}_{s}' for s in ('spring', 'summer') for b in CFG['sentinel2']['bands']]


def apply_mmu(l1, l2):
    ch = (l1 != l2) & (l1 > 0) & (l2 > 0)
    cc, n = ndimage.label(ch, structure=np.ones((3, 3)))
    small = (np.bincount(cc.ravel()) < MMU)[cc] & ch
    out = l2.copy(); out[small] = l1[small]
    return out


def cva_magnitude():
    a1, _, n = read(f'{PROC}/S2_{y1}_features.tif'); a2 = read(f'{PROC}/S2_{y2}_features.tif')[0]
    ix = [n.index(b) for b in CVA_BANDS]
    valid = (a1[ix] > -32768).all(0) & (a2[ix] > -32768).all(0)
    d = (a2[ix].astype(np.float32) - a1[ix]) / 10000
    return np.sqrt((d ** 2).sum(0)), valid


def transition(l1, l2):
    v = (l1 > 0) & (l2 > 0)
    m = np.zeros((5, 5))
    np.add.at(m, (l1[v] - 1, l2[v] - 1), 1)
    names = [CLASSES[c] for c in CLASSES]
    return pd.DataFrame(m * PIX_HA, index=names, columns=names)


def class_area(tm):
    a = pd.DataFrame({f'{y1}_ha': tm.sum(axis=1).values, f'{y2}_ha': tm.sum(axis=0).values}, index=tm.index)
    a['net_ha'] = a[f'{y2}_ha'] - a[f'{y1}_ha']
    a['net_pct'] = 100 * a.net_ha / a[f'{y1}_ha']
    return a


STRATA = {1: 'Stable water', 2: 'Stable tree cover', 3: 'Stable low vegetation', 4: 'Stable built-up',
          5: 'Stable bare/sparse', 6: 'CPCC change to built-up', 7: 'CPCC other change',
          8: 'PCC change without CVA support'}
ALLOC = {1: 30, 2: 50, 3: 90, 4: 70, 5: 50, 6: 70, 7: 70, 8: 70}

if __name__ == '__main__':
    prof = read(f'{MAPS}/lulc_{y1}_rf.tif')[1]
    raw1 = read(f'{MAPS}/lulc_{y1}_rf_raw.tif')[0][0]; raw2 = read(f'{MAPS}/lulc_{y2}_rf_raw.tif')[0][0]
    f1 = read(f'{MAPS}/lulc_{y1}_rf.tif')[0][0]; f2 = read(f'{MAPS}/lulc_{y2}_rf.tif')[0][0]
    pcc2 = apply_mmu(f1, f2)
    M, valid = cva_magnitude()
    rs = np.random.default_rng(SEED).choice(np.flatnonzero(valid), 500000, replace=False)
    t = float(threshold_otsu(M.ravel()[rs]))
    spec = (M > t) & valid
    cpcc2 = apply_mmu(f1, np.where(spec, f2, f1).astype(np.uint8))
    write(f'{MAPS}/lulc_{y2}_pcc.tif', pcc2, prof, 'uint8', 0)
    write(f'{MAPS}/lulc_{y2}_cpcc.tif', cpcc2, prof, 'uint8', 0)
    write(f'{MAPS}/cva_magnitude.tif', np.round(M * 1000).astype(np.uint16), prof, 'uint16', 0)
    for tag, l2 in (('pcc', pcc2), ('cpcc', cpcc2)):
        code = np.where((f1 > 0) & (l2 > 0), f1.astype(np.uint16) * 10 + l2, 0)
        write(f'{MAPS}/change_{tag}_{y1}_{y2}.tif', code, prof, 'uint16', 0)
        tm = transition(f1, l2); tm.to_csv(f'{TABS}/transition_matrix_{tag}_ha.csv')
        class_area(tm).to_csv(f'{TABS}/class_area_map_{tag}.csv')
        print(tag); print(tm.round(1))
    ch = lambda a, b: float(((a != b) & (a > 0) & (b > 0)).sum() * PIX_HA)
    info = {'cva_otsu_threshold': t, 'cva_fraction_above': float(spec[valid].mean()),
            'change_ha_raw_maps': ch(raw1, raw2), 'change_ha_majority': ch(f1, f2),
            'change_ha_pcc_mmu': ch(f1, pcc2), 'change_ha_cpcc_mmu': ch(f1, cpcc2),
            'total_ha': float(((f1 > 0) & (pcc2 > 0)).sum() * PIX_HA)}
    json.dump(info, open(f'{TABS}/change_area_postprocessing.json', 'w'), indent=1)
    print(info)

    # strata: stable classes, CPCC change split by built-up gain, PCC change rejected by CVA
    s = np.zeros(f1.shape, np.uint8)
    stable = f1 == pcc2
    for c in CLASSES:
        s[stable & (f1 == c)] = c
    cch = cpcc2 != f1
    s[cch & (cpcc2 == 4)] = 6
    s[cch & (cpcc2 != 4)] = 7
    s[~stable & ~cch] = 8
    s[(f1 == 0) | (pcc2 == 0)] = 0
    write(f'{MAPS}/strata.tif', s, prof, 'uint8', 0)
    rng = np.random.default_rng(SEED + 1)
    T = prof['transform']; to_ll = Transformer.from_crs(CFG['project']['crs'], 'EPSG:4326', always_xy=True)
    Nh = {h: int((s == h).sum()) for h in STRATA}
    recs = []
    for h, n in ALLOC.items():
        r, c = np.nonzero(s == h)
        k = rng.choice(len(r), size=min(n, len(r)), replace=False)
        recs.append(pd.DataFrame({'row': r[k], 'col': c[k], 'stratum': h}))
    sp = pd.concat(recs, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sp['pid'] = np.arange(1, len(sp) + 1)
    sp['x'] = T.c + (sp.col + 0.5) * T.a; sp['y'] = T.f + (sp.row + 0.5) * T.e
    sp['lon'], sp['lat'] = to_ll.transform(sp.x.values, sp.y.values)
    sp.to_csv(f'{SAMP}/validation_points.csv', index=False)
    al = pd.DataFrame({'stratum': [STRATA[h] for h in STRATA], 'N_pixels': [Nh[h] for h in STRATA],
                       'area_ha': [Nh[h] * PIX_HA for h in STRATA], 'n_sample': [ALLOC[h] for h in STRATA]},
                      index=pd.Index(list(STRATA), name='h'))
    al.to_csv(f'{TABS}/strata_allocation.csv'); print(al)
