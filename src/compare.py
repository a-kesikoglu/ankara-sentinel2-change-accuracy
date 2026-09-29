"""Design-based paired differences in overall accuracy between classifier and feature-set
configurations evaluated on the same stratified reference sample."""
import json, ast
import numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import *
from estimators import Design
from classify import FS, load_features, training_table, rf

y1, y2 = YEARS
sp = pd.read_csv(f'{SAMP}/validation_points.csv').merge(pd.read_csv(f'{SAMP}/reference_labels.csv'), on='pid')
al = pd.read_csv(f'{TABS}/strata_allocation.csv', index_col=0)
D = Design(sp.stratum.values, al.N_pixels.to_dict())
exp = pd.read_csv(f'{TABS}/experiments_classifier_featureset.csv')
d = training_table()
X = {}
for y in YEARS:
    arr, _, names = load_features(y)
    X[y] = pd.DataFrame(arr[:, sp.row.values, sp.col.values].T.astype(np.float32) / 10000, columns=names)

pred = {}
for fs in ('S-B', 'SS-B', 'SS-BI'):
    for c in ('RF', 'SVM'):
        cols = FS[fs]
        if c == 'RF':
            m = rf()
        else:
            p = ast.literal_eval(exp[(exp.feature_set == fs) & (exp.classifier == 'SVM')].params.iloc[0])
            m = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced', C=p['svc__C'], gamma=p['svc__gamma']))
        m.fit(d[cols].values, d.class_id.values)
        pred[(c, fs)] = (m.predict(X[y1][cols].values), m.predict(X[y2][cols].values))

r1, r2 = sp.ref_2018.values, sp.ref_2025.values
out = {}
for a, b in ((('SVM', 'SS-BI'), ('RF', 'SS-BI')), (('SVM', 'SS-B'), ('RF', 'SS-B')), (('SVM', 'S-B'), ('RF', 'S-B')),
             (('RF', 'SS-B'), ('RF', 'S-B')), (('SVM', 'SS-B'), ('SVM', 'S-B')), (('RF', 'SS-BI'), ('RF', 'SS-B'))):
    rec = {}
    for i, (y, ref) in enumerate(((y1, r1), (y2, r2))):
        diff, se = D.proportion((pred[a][i] == ref).astype(float) - (pred[b][i] == ref).astype(float))
        rec[str(y)] = {'diff_pp': 100 * diff, 'ci95_pp': 196 * se, 'z': diff / se if se > 0 else np.nan}
    out[f'{a[0]} {a[1]} - {b[0]} {b[1]}'] = rec
json.dump(out, open(f'{TABS}/paired_oa_differences.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
