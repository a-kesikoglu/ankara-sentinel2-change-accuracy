"""Design-based accuracy assessment, area estimation and method comparison using the
stratified reference sample (reference labels interpreted for 2018 and 2025)."""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from common import *
from estimators import Design, accuracy, area, Z95
from classify import FS, load_features, training_table, rf

y1, y2 = YEARS
K = list(CLASSES)
PIX_HA = CFG['project']['pixel_size_m'] ** 2 / 1e4

sp = pd.read_csv(f'{SAMP}/validation_points.csv')
lab = pd.read_csv(f'{SAMP}/reference_labels.csv')
sp = sp.merge(lab, on='pid')
al = pd.read_csv(f'{TABS}/strata_allocation.csv', index_col=0)
D = Design(sp.stratum.values, al.N_pixels.to_dict())
r1, r2 = sp.ref_2018.values, sp.ref_2025.values
rc = r1 != r2


def at(path):
    return read(path)[0][0][sp.row.values, sp.col.values]


maps = {'RF_2018': at(f'{MAPS}/lulc_{y1}_rf.tif'), 'RF_PCC_2025': at(f'{MAPS}/lulc_{y2}_pcc.tif'),
        'RF_CPCC_2025': at(f'{MAPS}/lulc_{y2}_cpcc.tif'),
        'DW_2018': DW_MAP[at(f'{PROC}/DW_{y1}_label.tif')], 'DW_2025': DW_MAP[at(f'{PROC}/DW_{y2}_label.tif')]}
wc = remap_wc(at(f'{PROC}/WC_2021.tif'))


def fmt(p, se, pct=True):
    f = 100 if pct else 1
    return f'{p * f:.1f} ± {Z95 * se * f:.1f}'


# ---- per-date thematic accuracy -------------------------------------------------------
rows, ems = [], {}
for name, ref in (('RF_2018', r1), ('RF_PCC_2025', r2), ('RF_CPCC_2025', r2), ('DW_2018', r1), ('DW_2025', r2)):
    oa, tab, em = accuracy(D, maps[name], ref, K)
    tab.insert(0, 'map', name); tab['OA'] = oa[0]; tab['OA_se'] = oa[1]
    rows.append(tab); ems[name] = em
acc = pd.concat(rows); acc['class_name'] = acc['class'].map(CLASSES)
acc.to_csv(f'{TABS}/accuracy_by_class.csv', index=False)
for k, em in ems.items():
    pd.DataFrame(em, index=[CLASSES[c] for c in K], columns=[CLASSES[c] for c in K]).to_csv(f'{TABS}/error_matrix_{k}.csv')

# sample-count error matrices (map rows, reference columns) for the publication tables
for name, ref in (('RF_2018', r1), ('RF_PCC_2025', r2), ('RF_CPCC_2025', r2), ('DW_2018', r1), ('DW_2025', r2)):
    pd.crosstab(pd.Series(maps[name], name='map'), pd.Series(ref, name='ref')).reindex(index=K, columns=K, fill_value=0) \
        .to_csv(f'{TABS}/error_matrix_counts_{name}.csv')

# ---- change detection accuracy ----------------------------------------------------------
chg = {'RF PCC': maps['RF_2018'] != maps['RF_PCC_2025'], 'RF CPCC': maps['RF_2018'] != maps['RF_CPCC_2025'],
       'DW PCC': maps['DW_2018'] != maps['DW_2025']}
gain_ref = (r1 != 4) & (r2 == 4)
gain = {'RF PCC': (maps['RF_2018'] != 4) & (maps['RF_PCC_2025'] == 4),
        'RF CPCC': (maps['RF_2018'] != 4) & (maps['RF_CPCC_2025'] == 4),
        'DW PCC': (maps['DW_2018'] != 4) & (maps['DW_2025'] == 4)}
full = {'RF PCC': (maps['RF_2018'] == r1) & (maps['RF_PCC_2025'] == r2),
        'RF CPCC': (maps['RF_2018'] == r1) & (maps['RF_CPCC_2025'] == r2),
        'DW PCC': (maps['DW_2018'] == r1) & (maps['DW_2025'] == r2)}
crow = []
for m in chg:
    rec = {'method': m}
    for tag, mp, rf_ in (('change', chg[m], rc), ('builtgain', gain[m], gain_ref)):
        ua = D.ratio(mp & rf_, mp); pa = D.ratio(mp & rf_, rf_)
        rec.update({f'{tag}_UA': ua[0], f'{tag}_UA_se': ua[1], f'{tag}_PA': pa[0], f'{tag}_PA_se': pa[1],
                    f'{tag}_F1': 2 * ua[0] * pa[0] / (ua[0] + pa[0]) if ua[0] + pa[0] > 0 else np.nan})
    oa = D.proportion(chg[m] == rc); rec['change_OA'], rec['change_OA_se'] = oa
    ft = D.proportion(full[m]); rec['trajectory_OA'], rec['trajectory_OA_se'] = ft
    mp_area = {'RF PCC': 'change_pcc', 'RF CPCC': 'change_cpcc'}.get(m)
    crow.append(rec)
cacc = pd.DataFrame(crow); cacc.to_csv(f'{TABS}/change_accuracy.csv', index=False)

# paired differences in change/no-change overall accuracy
diffs = {}
for a, b in (('RF CPCC', 'RF PCC'), ('RF CPCC', 'DW PCC'), ('RF PCC', 'DW PCC')):
    p, se = D.proportion((chg[a] == rc).astype(float) - (chg[b] == rc).astype(float))
    diffs[f'{a} - {b}'] = {'diff': p, 'se': se, 'z': p / se if se > 0 else np.nan}
json.dump(diffs, open(f'{TABS}/change_oa_differences.json', 'w'), indent=1)

# ---- area estimation --------------------------------------------------------------------
full_map = {y1: read(f'{MAPS}/lulc_{y1}_rf.tif')[0][0], y2: read(f'{MAPS}/lulc_{y2}_cpcc.tif')[0][0]}
arow = []
for y, ref in ((y1, r1), (y2, r2)):
    for k in K:
        a, se = area(D, ref, k)
        arow.append({'year': y, 'class': CLASSES[k], 'map_ha': float((full_map[y] == k).sum() * PIX_HA),
                     'est_ha': a * PIX_HA, 'ci95_ha': Z95 * se * PIX_HA})
pcc2 = read(f'{MAPS}/lulc_{y2}_pcc.tif')[0][0]
vm = (full_map[y1] > 0) & (full_map[y2] > 0) & (pcc2 > 0)
for tag, mref in (('Total change', rc), ('Gain of built-up', gain_ref)):
    p, se = D.proportion(mref)
    rec = {'year': f'{y1}-{y2}', 'class': tag, 'est_ha': p * D.N * PIX_HA, 'ci95_ha': Z95 * se * D.N * PIX_HA}
    if tag == 'Total change':
        rec['map_ha_pcc'] = float(((full_map[y1] != pcc2) & vm).sum() * PIX_HA)
        rec['map_ha'] = float(((full_map[y1] != full_map[y2]) & vm).sum() * PIX_HA)
    else:
        rec['map_ha_pcc'] = float(((full_map[y1] != 4) & (pcc2 == 4) & vm).sum() * PIX_HA)
        rec['map_ha'] = float(((full_map[y1] != 4) & (full_map[y2] == 4) & vm).sum() * PIX_HA)
    arow.append(rec)

# major transitions: map areas (PCC, CPCC) and design-based estimates with 95% CI
trows = []
for i in K:
    for j in K:
        if i == j:
            continue
        p, se = D.proportion((r1 == i) & (r2 == j))
        mp = float(((full_map[y1] == i) & (pcc2 == j) & vm).sum() * PIX_HA)
        mc = float(((full_map[y1] == i) & (full_map[y2] == j) & vm).sum() * PIX_HA)
        trows.append({'from': CLASSES[i], 'to': CLASSES[j], 'n_ref': int(((r1 == i) & (r2 == j)).sum()),
                      'map_pcc_ha': mp, 'map_cpcc_ha': mc, 'est_ha': p * D.N * PIX_HA, 'ci95_ha': Z95 * se * D.N * PIX_HA})
pd.DataFrame(trows).sort_values('est_ha', ascending=False).to_csv(f'{TABS}/transitions_estimated.csv', index=False)
areas = pd.DataFrame(arow); areas.to_csv(f'{TABS}/area_estimates.csv', index=False)

# reference transition matrix (estimated area, ha)
rtm = np.array([[D.proportion((r1 == i) & (r2 == j))[0] * D.N * PIX_HA for j in K] for i in K])
pd.DataFrame(rtm, index=[CLASSES[c] for c in K], columns=[CLASSES[c] for c in K]).to_csv(f'{TABS}/transition_reference_estimated_ha.csv')

# ---- classifier and feature-set experiments (per-pixel predictions, no post-processing) --
d = training_table()
Xs = {}
for y in YEARS:
    arr, _, names = load_features(y)
    Xs[y] = pd.DataFrame(arr[:, sp.row.values, sp.col.values].T.astype(np.float32) / 10000, columns=names)
exp = []
for fs, cols in FS.items():
    for clf_name in ('RF', 'SVM'):
        if clf_name == 'RF':
            clf = rf()
        else:
            gs = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced')),
                              {'svc__C': CFG['classification']['svm']['C_grid'],
                               'svc__gamma': CFG['classification']['svm']['gamma_grid']},
                              cv=GroupKFold(5), n_jobs=-1)
            gs.fit(d[cols].values, d.class_id.values, groups=d.pid.values)
            clf = gs.best_estimator_
        clf.fit(d[cols].values, d.class_id.values)
        p1 = clf.predict(Xs[y1][cols].values); p2 = clf.predict(Xs[y2][cols].values)
        o1 = D.proportion(p1 == r1); o2 = D.proportion(p2 == r2)
        mc = p1 != p2
        ua = D.ratio(mc & rc, mc)[0]; pa = D.ratio(mc & rc, rc)[0]
        exp.append({'feature_set': fs, 'n_features': len(cols), 'classifier': clf_name,
                    'OA_2018': o1[0], 'OA_2018_se': o1[1], 'OA_2025': o2[0], 'OA_2025_se': o2[1],
                    'change_UA': ua, 'change_PA': pa, 'change_F1': 2 * ua * pa / (ua + pa),
                    'params': str(gs.best_params_) if clf_name == 'SVM' else ''})
        print(exp[-1])
pd.DataFrame(exp).to_csv(f'{TABS}/experiments_classifier_featureset.csv', index=False)

# ---- agreement with open products (full map, pixel level) ---------------------------------
dwf = {y: DW_MAP[read(f'{PROC}/DW_{y}_label.tif')[0][0]] for y in YEARS}
wcf = remap_wc(read(f'{PROC}/WC_2021.tif')[0][0])
agree = {}
for name, a, b in ((f'RF {y1} vs DW {y1}', full_map[y1], dwf[y1]), (f'RF {y2} vs DW {y2}', full_map[y2], dwf[y2]),
                   (f'RF {y1} vs WC 2021', full_map[y1], wcf), (f'RF {y2} vs WC 2021', full_map[y2], wcf)):
    v = (a > 0) & (b > 0)
    agree[name] = {'overall_agreement': float((a[v] == b[v]).mean()),
                   'per_class_jaccard': {CLASSES[k]: float(((a == k) & (b == k) & v).sum() / max((((a == k) | (b == k)) & v).sum(), 1))
                                         for k in K}}
json.dump(agree, open(f'{TABS}/agreement_open_products.json', 'w'), indent=1)
print(acc.groupby('map')[['OA', 'OA_se']].first()); print(cacc.round(3).T); print(areas.round(1)); print(json.dumps(diffs, indent=1))
