"""Sensitivity of classification and change accuracy to the s2cloudless probability threshold.
Features (SS-BI set) are sampled in Earth Engine at the training and validation locations for
thresholds of 30, 40 and 50%, the RF is retrained for each threshold and evaluated with the
design-based estimators on the same reference sample."""
import json
import numpy as np, pandas as pd, ee
import gee_download as g
from common import *
from estimators import Design, Z95
from classify import FS, rf

y1, y2 = YEARS
tp = pd.read_csv(f'{SAMP}/training_points.csv')
sp = pd.read_csv(f'{SAMP}/validation_points.csv').merge(pd.read_csv(f'{SAMP}/reference_labels.csv'), on='pid')
al = pd.read_csv(f'{TABS}/strata_allocation.csv', index_col=0)
D = Design(sp.stratum.values, al.N_pixels.to_dict())
r1, r2 = sp.ref_2018.values, sp.ref_2025.values
cols = FS['SS-BI']
PROJ = ee.Projection(CFG['project']['crs']).atScale(10)


def points(df, prefix):
    feats = [ee.Feature(ee.Geometry.Point([float(x), float(y)], CFG['project']['crs']), {'pt': f'{prefix}{i}'})
             for i, (x, y) in enumerate(zip(df.x, df.y))]
    return ee.FeatureCollection(feats)


FC = points(tp, 't').merge(points(sp, 'v'))


def sample(thr, key):
    g.CFG['sentinel2']['cloud_probability_threshold'] = thr
    feat, cnt = g.features(CFG['periods'][key])
    img = feat.addBands(cnt)
    out = img.sampleRegions(collection=FC, properties=['pt'], projection=PROJ, geometries=False)
    rows = out.getInfo()['features']
    df = pd.DataFrame([r['properties'] for r in rows]).set_index('pt')
    med = cnt.reduceRegion(ee.Reducer.median(), g.AOI, 30, maxPixels=1e9, bestEffort=True).getInfo()
    return df, med


res = []
for thr in (30, 40, 50):
    data, meds = {}, {}
    for key, y in (('t1', y1), ('t2', y2)):
        data[y], meds[y] = sample(thr, key)
        print(thr, y, data[y].shape, meds[y], flush=True)
    Xtr = pd.concat([data[y].reindex([f't{i}' for i in range(len(tp))])[cols].assign(cls=tp.class_id.values) for y in YEARS])
    Xtr = Xtr.dropna()
    Xtr = Xtr[(Xtr[cols] > -30000).all(axis=1)]
    m = rf().fit(Xtr[cols].values / 10000, Xtr.cls.values)
    pv = {}
    for y in YEARS:
        v = data[y].reindex([f'v{i}' for i in range(len(sp))])[cols].fillna(0).values / 10000
        pv[y] = m.predict(v)
    o1 = D.proportion(pv[y1] == r1); o2 = D.proportion(pv[y2] == r2)
    mc, rc = pv[y1] != pv[y2], r1 != r2
    ua = D.ratio(mc & rc, mc); pa = D.ratio(mc & rc, rc)
    res.append({'threshold': thr, 'n_train': len(Xtr),
                'median_obs_spring_2018': meds[y1].get('n_spring'), 'median_obs_summer_2018': meds[y1].get('n_summer'),
                'median_obs_spring_2025': meds[y2].get('n_spring'), 'median_obs_summer_2025': meds[y2].get('n_summer'),
                'OA_2018': o1[0], 'OA_2018_ci': Z95 * o1[1], 'OA_2025': o2[0], 'OA_2025_ci': Z95 * o2[1],
                'change_UA': ua[0], 'change_UA_ci': Z95 * ua[1], 'change_PA': pa[0], 'change_PA_ci': Z95 * pa[1]})
    print(res[-1], flush=True)
pd.DataFrame(res).to_csv(f'{TABS}/cloud_threshold_sensitivity.csv', index=False)
