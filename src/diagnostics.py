"""Robustness and diagnostic analyses requested in revision:
reference-subset sensitivity, spatial separation of training and validation locations, CVA magnitude
by reference transition, grouped permutation importance (design-based) and the supplementary sample table."""
import json, joblib
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from common import *
from estimators import Design, Z95
from classify import FS, load_features

y1, y2 = YEARS
PIX_HA = CFG['project']['pixel_size_m'] ** 2 / 1e4
sp0 = pd.read_csv(f'{SAMP}/validation_points.csv').merge(pd.read_csv(f'{SAMP}/reference_labels.csv'), on='pid')
al = pd.read_csv(f'{TABS}/strata_allocation.csv', index_col=0)
Nh = al.N_pixels.to_dict()


def at(path, df):
    return read(path)[0][0][df.row.values, df.col.values]


sp0['map_2018'] = at(f'{MAPS}/lulc_{y1}_rf.tif', sp0)
sp0['map_2025_pcc'] = at(f'{MAPS}/lulc_{y2}_pcc.tif', sp0)
sp0['map_2025_cpcc'] = at(f'{MAPS}/lulc_{y2}_cpcc.tif', sp0)
sp0['dw_2018'] = DW_MAP[at(f'{PROC}/DW_{y1}_label.tif', sp0)]
sp0['dw_2025'] = DW_MAP[at(f'{PROC}/DW_{y2}_label.tif', sp0)]
sp0['cva'] = at(f'{MAPS}/cva_magnitude.tif', sp0) / 1000
d1 = pd.to_datetime(sp0.vhr_date_t1, unit='ms'); d2 = pd.to_datetime(sp0.vhr_date_t2, unit='ms')
sp0['vhr_date_2018'] = d1.dt.date; sp0['vhr_date_2025'] = d2.dt.date
sp0['vhr_matched'] = (d1.dt.year == y1) & (d2.dt.year == y2)


# ---- 1. sensitivity of main estimates to reference subsets -------------------------------
def estimates(sp):
    D = Design(sp.stratum.values, Nh)
    r1, r2 = sp.ref_2018.values, sp.ref_2025.values
    m1, m2 = sp.map_2018.values, sp.map_2025_pcc.values
    rc, mc = r1 != r2, m1 != m2
    out = {'n': len(sp)}
    for k, v in (('OA_2018', m1 == r1), ('OA_2025', m2 == r2)):
        p, se = D.proportion(v); out[k] = (100 * p, 100 * Z95 * se)
    ua = D.ratio(mc & rc, mc); pa = D.ratio(mc & rc, rc)
    out['change_UA'] = (100 * ua[0], 100 * Z95 * ua[1]); out['change_PA'] = (100 * pa[0], 100 * Z95 * pa[1])
    p, se = D.proportion(rc); out['total_change_ha'] = (p * D.N * PIX_HA, Z95 * se * D.N * PIX_HA)
    p, se = D.proportion((r1 != 4) & (r2 == 4)); out['builtup_gain_ha'] = (p * D.N * PIX_HA, Z95 * se * D.N * PIX_HA)
    p, se = D.proportion((r2 == 4).astype(float) - (r1 == 4)); out['net_builtup_ha'] = (p * D.N * PIX_HA, Z95 * se * D.N * PIX_HA)
    return out


sens = {'All samples': estimates(sp0),
        'Confidence medium or high': estimates(sp0[sp0.confidence >= 2]),
        'VHR acquired in target years': estimates(sp0[sp0.vhr_matched]),
        'Both restrictions': estimates(sp0[(sp0.confidence >= 2) & sp0.vhr_matched])}
json.dump(sens, open(f'{TABS}/sensitivity_reference_subsets.json', 'w'), indent=1)
print(pd.DataFrame(sens).T)

# ---- 2. spatial separation of validation and training locations ---------------------------
tp = pd.read_csv(f'{SAMP}/training_points.csv')
dist, _ = cKDTree(tp[['x', 'y']].values).query(sp0[['x', 'y']].values)
sp0['dist_training_m'] = dist
sep = {'median_m': float(np.median(dist)), 'p10_m': float(np.percentile(dist, 10)), 'min_m': float(dist.min()),
       'n_within_30m': int((dist <= 30).sum()), 'n_within_100m': int((dist <= 100).sum()), 'n_identical': int((dist < 1).sum())}
sens['Farther than 100 m from training'] = estimates(sp0[sp0.dist_training_m > 100])
json.dump(sens, open(f'{TABS}/sensitivity_reference_subsets.json', 'w'), indent=1)
json.dump(sep, open(f'{TABS}/spatial_separation.json', 'w'), indent=1)
print(sep); print(pd.DataFrame(sens).T[['n', 'OA_2018', 'OA_2025', 'change_UA', 'total_change_ha']])

# ---- 3. CVA magnitude by reference transition --------------------------------------------
def ttype(a, b):
    if a == b:
        return 'No change'
    lab = f'{CLASSES[a]} → {CLASSES[b]}'
    return lab if lab in ('Low vegetation → Built-up', 'Low vegetation → Bare/sparse', 'Bare/sparse → Low vegetation',
                          'Bare/sparse → Built-up') else 'Other change'


sp0['ref_transition'] = [ttype(a, b) for a, b in zip(sp0.ref_2018, sp0.ref_2025)]
thr = json.load(open(f'{TABS}/change_area_postprocessing.json'))['cva_otsu_threshold']
cv = sp0.groupby('ref_transition').cva.agg(['count', 'median', lambda s: (s > thr).mean()])
cv.columns = ['n', 'median_magnitude', 'share_above_otsu']
cv.to_csv(f'{TABS}/cva_by_reference_transition.csv'); print(cv.round(3))

# ---- 4. grouped permutation importance on the reference sample ----------------------------
model = joblib.load(f'{MAPS}/rf_final.joblib')
feats = FS['SS-BI-T']
X = {}
for y in YEARS:
    arr, _, names = load_features(y)
    X[y] = pd.DataFrame(arr[:, sp0.row.values, sp0.col.values].T.astype(np.float32) / 10000, columns=names)[feats]
D = Design(sp0.stratum.values, Nh)
refs = {y1: sp0.ref_2018.values, y2: sp0.ref_2025.values}
base = {y: D.proportion(model.predict(X[y].values) == refs[y])[0] for y in YEARS}
bands = CFG['sentinel2']['bands']; idx = CFG['sentinel2']['indices']
groups = {'Spring bands': [f'{b}_spring' for b in bands], 'Summer bands': [f'{b}_summer' for b in bands],
          'Spring indices': [f'{i}_spring' for i in idx], 'Summer indices': [f'{i}_summer' for i in idx],
          'Texture': [f for f in feats if f.startswith('SD5_')]}
rng = np.random.default_rng(SEED)
rows = []
for gname, gcols in groups.items():
    drops = []
    for rep in range(20):
        dd = []
        for y in YEARS:
            Xp = X[y].copy()
            perm = rng.permutation(len(Xp))
            Xp[gcols] = Xp[gcols].values[perm]
            dd.append(base[y] - D.proportion(model.predict(Xp.values) == refs[y])[0])
        drops.append(np.mean(dd))
    rows.append({'group': gname, 'n_features': len(gcols), 'mean_drop_pp': 100 * np.mean(drops), 'sd_drop_pp': 100 * np.std(drops)})
pi = pd.DataFrame(rows); pi.to_csv(f'{TABS}/permutation_importance_grouped.csv', index=False); print(pi.round(2))

# ---- 5. supplementary sample table ---------------------------------------------------------
rel = {'2018': 'Wayback 2019-12-12 (release 4756)', '2025': 'Wayback 2026-08-05 (release 26334)'}
S = sp0[['pid', 'lon', 'lat', 'x', 'y', 'stratum', 'ref_2018', 'ref_2025', 'confidence', 'vhr_date_2018', 'vhr_res_t1',
         'vhr_date_2025', 'vhr_res_t2', 'vhr_matched', 'map_2018', 'map_2025_pcc', 'map_2025_cpcc', 'dw_2018', 'dw_2025',
         'cva', 'dist_training_m']].copy()
S.insert(9, 'vhr_release_2018', rel['2018']); S.insert(12, 'vhr_release_2025', rel['2025'])
S = S.rename(columns={'vhr_res_t1': 'vhr_res_2018_m', 'vhr_res_t2': 'vhr_res_2025_m', 'x': 'x_utm36n', 'y': 'y_utm36n'})
S.sort_values('pid').to_csv(f'{TABS}/S_reference_sample.csv', index=False)
print('matched', int(sp0.vhr_matched.sum()))
