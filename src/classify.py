"""Train the Random Forest on pooled stable samples (both years) and classify both composites.
Also writes 3x3 majority-filtered maps and the post-classification change products."""
import json, joblib
import numpy as np, pandas as pd
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from common import *

FS = {
    'S-B': [f'{b}_summer' for b in CFG['sentinel2']['bands']],
    'S-BI': [f'{b}_summer' for b in CFG['sentinel2']['bands']] + [f'{i}_summer' for i in CFG['sentinel2']['indices']],
    'SS-B': [f'{b}_{s}' for s in ('spring', 'summer') for b in CFG['sentinel2']['bands']],
}
FS['SS-BI'] = FS['SS-B'] + [f'{i}_{s}' for s in ('spring', 'summer') for i in CFG['sentinel2']['indices']]
TEXTURE = ['B8_summer', 'NDVI_spring', 'NDBI_summer']
FS['SS-BI-T'] = FS['SS-BI'] + [f'SD5_{b}' for b in TEXTURE]


def focal_sd(x, size=5):
    x = x.astype(np.float32)
    m = ndimage.uniform_filter(x, size)
    return np.sqrt(np.maximum(ndimage.uniform_filter(x * x, size) - m * m, 0))


def load_features(year):
    """Composite bands and indices plus 5 x 5 local standard deviation texture layers."""
    arr, prof, names = read(f'{PROC}/S2_{year}_features.tif')
    tex = [np.round(focal_sd(arr[names.index(b)])).astype(np.int16) for b in TEXTURE]
    return np.concatenate([arr, np.stack(tex)]), prof, names + [f'SD5_{b}' for b in TEXTURE]


def training_table():
    tp = pd.read_csv(f'{SAMP}/training_points.csv')
    out = []
    for y in YEARS:
        arr, _, names = load_features(y)
        X = arr[:, tp.row.values, tp.col.values].T.astype(np.float32) / 10000
        d = pd.DataFrame(X, columns=names)
        d['class_id'] = tp.class_id.values; d['year'] = y; d['pid'] = np.arange(len(tp))
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    return d[(d[FS['SS-BI']] > -3).all(axis=1)]


def rf():
    p = CFG['classification']['rf']
    return RandomForestClassifier(n_estimators=p['n_estimators'], max_features=p['max_features'],
                                  min_samples_leaf=p['min_samples_leaf'], class_weight=p['class_weight'],
                                  oob_score=True, n_jobs=-1, random_state=SEED)


def majority3(lab):
    counts = np.stack([ndimage.uniform_filter((lab == c).astype(np.float32), 3) for c in CLASSES])
    out = (np.argmax(counts, 0) + 1).astype(np.uint8)
    # keep original label where it ties with the winner
    best = counts.max(0)
    orig = np.take_along_axis(counts, (np.clip(lab, 1, 5) - 1)[None].astype(int), 0)[0]
    out[orig >= best - 1e-6] = lab[orig >= best - 1e-6]
    out[lab == 0] = 0
    return out


if __name__ == '__main__':
    d = training_table()
    feats = FS['SS-BI-T']
    model = rf().fit(d[feats].values, d.class_id.values)
    joblib.dump(model, f'{MAPS}/rf_final.joblib', compress=3)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    imp.to_csv(f'{TABS}/feature_importance_mdi.csv', header=['mdi'])
    json.dump({'oob_score': model.oob_score_, 'n_train': len(d),
               'n_per_class': d.class_id.value_counts().sort_index().to_dict()},
              open(f'{TABS}/rf_training_info.json', 'w'), indent=1)
    print('OOB', model.oob_score_)
    for y in YEARS:
        arr, prof, names = load_features(y)
        ix = [names.index(f) for f in feats]
        X = arr[ix].reshape(len(ix), -1).T.astype(np.float32) / 10000
        valid = (arr[ix] > -32768).all(0).ravel()
        lab = np.zeros(X.shape[0], np.uint8); pmax = np.zeros(X.shape[0], np.uint8)
        for s in range(0, X.shape[0], 500000):
            sl = slice(s, s + 500000); v = valid[sl]
            pr = model.predict_proba(X[sl][v])
            lab[sl][v] = model.classes_[pr.argmax(1)]
            pmax[sl][v] = np.round(pr.max(1) * 100)
        lab = lab.reshape(arr.shape[1:]); pmax = pmax.reshape(arr.shape[1:])
        p = prof.copy(); p.update(tiled=True)
        write(f'{MAPS}/lulc_{y}_rf_raw.tif', lab, p, 'uint8', 0)
        write(f'{MAPS}/lulc_{y}_rf_pmax.tif', pmax, p, 'uint8', 0)
        write(f'{MAPS}/lulc_{y}_rf.tif', majority3(lab), p, 'uint8', 0)
        print('classified', y, np.bincount(lab.ravel(), minlength=6))
