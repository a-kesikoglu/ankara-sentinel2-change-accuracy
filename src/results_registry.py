# -*- coding: utf-8 -*-
"""Single source of every number quoted in the manuscript. Text placeholders <<key>> and all
tables are filled from here, so abstract, text and tables cannot disagree."""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, 'outputs', 'tables')
Z = 1.96
CLS = ['Water', 'Tree cover', 'Low vegetation', 'Built-up', 'Bare/sparse']


def f0(v):
    return f'{v:,.0f}'


def f1(v):
    return f'{v:,.1f}'


def pct(p, se=None, d=1):
    """proportion (0-1) and its standard error -> 'xx.x ± y.y'"""
    s = f'{100 * p:.{d}f}'
    return s if se is None else f'{s} ± {100 * Z * se:.{d}f}'


def ha(v, ci=None):
    return f0(v) if ci is None else f'{f0(v)} ± {f0(ci)}'


def load(name):
    p = os.path.join(T, name)
    return json.load(open(p)) if name.endswith('.json') else pd.read_csv(p)


ACC = load('accuracy_by_class.csv')
CH = load('change_accuracy.csv').set_index('method')
AR = load('area_estimates.csv')
TR = load('transitions_estimated.csv')
MP = load('class_area_map_pcc.csv').set_index('Unnamed: 0')
MPC = load('class_area_map_cpcc.csv').set_index('Unnamed: 0')
AL = load('strata_allocation.csv').set_index('h')
POST = load('change_area_postprocessing.json')
TRPCC = load('transition_matrix_pcc_ha.csv').set_index('Unnamed: 0')
EXP = load('experiments_classifier_featureset.csv')
PAIR = load('paired_oa_differences.json')
DIFF = load('change_oa_differences.json')
AGR = load('agreement_open_products.json')
SENS = load('sensitivity_reference_subsets.json')
SEP = load('spatial_separation.json')
CVA = load('cva_by_reference_transition.csv').set_index('ref_transition')
PERM = load('permutation_importance_grouped.csv').set_index('group')
CLOUD = load('cloud_threshold_sensitivity.csv').set_index('threshold')
TRAIN = load('training_sample_counts.csv').set_index('Unnamed: 0')
INFO = load('rf_training_info.json')
SCN = load('scene_stats.json')
SMP = load('S_reference_sample.csv')
IMP = load('feature_importance_mdi.csv')


def acc(mapname, cls, col):
    r = ACC[(ACC['map'] == mapname) & (ACC['class_name'] == cls)].iloc[0]
    return r[col], r.get(col + '_se', None)


def oa(mapname):
    r = ACC[ACC['map'] == mapname].iloc[0]
    return r.OA, r.OA_se


def ar(year, cls):
    return AR[(AR.year.astype(str) == str(year)) & (AR['class'] == cls)].iloc[0]


def tr(a, b):
    return TR[(TR['from'] == a) & (TR['to'] == b)].iloc[0]


def dw_change_area():
    import rasterio
    lut = np.array([1, 2, 3, 3, 3, 3, 4, 5, 0])
    a = lut[rasterio.open(os.path.join(ROOT, 'data', 'processed', 'DW_2018_label.tif')).read(1)]
    b = lut[rasterio.open(os.path.join(ROOT, 'data', 'processed', 'DW_2025_label.tif')).read(1)]
    return float(((a != b) & (a > 0) & (b > 0)).sum() / 100)


N = {}
tot_ha = POST['total_ha']
N['area_km2'] = f'{tot_ha / 100:.1f}'
N['area_ha'] = f0(tot_ha)
# scenes and observations
for y in ('2018', '2025'):
    for s in ('spring', 'summer'):
        N[f'scenes_{s}_{y}'] = str(SCN[y][s]['n_scenes'])
cl40 = CLOUD.loc[40]
N['obs_spring_2018'] = f0(cl40.median_obs_spring_2018); N['obs_summer_2018'] = f0(cl40.median_obs_summer_2018)
N['obs_spring_2025'] = f0(cl40.median_obs_spring_2025); N['obs_summer_2025'] = f0(cl40.median_obs_summer_2025)
# training
for c, k in zip(CLS, ('water', 'tree', 'lowveg', 'built', 'bare')):
    N[f'train_{k}'] = f0(TRAIN.loc[c, 'training_points'])
N['n_train'] = f0(INFO['n_train']); N['oob'] = pct(INFO['oob_score'])
# single-date accuracy
for m, k in (('RF_2018', 'rf18'), ('RF_PCC_2025', 'rf25'), ('RF_CPCC_2025', 'cp25'), ('DW_2018', 'dw18'), ('DW_2025', 'dw25')):
    N[f'oa_{k}'] = pct(*oa(m)); N[f'oa_{k}_v'] = pct(oa(m)[0])
    for c, ck in zip(CLS, ('water', 'tree', 'lowveg', 'built', 'bare')):
        for col in ('UA', 'PA'):
            N[f'{col.lower()}_{ck}_{k}'] = pct(acc(m, c, col)[0])
N['oob_gap'] = f'{100 * (INFO["oob_score"] - oa("RF_2018")[0]):.0f}'
em = pd.read_csv(os.path.join(T, 'error_matrix_counts_RF_2018.csv'), index_col=0)
N['water_as_built_2018'] = str(int(em.iloc[0, 3])); N['n_map_water_2018'] = str(int(em.iloc[0].sum()))
# agreement
N['agr_dw18'] = pct(AGR['RF 2018 vs DW 2018']['overall_agreement']); N['agr_dw25'] = pct(AGR['RF 2025 vs DW 2025']['overall_agreement'])
N['agr_wc18'] = pct(AGR['RF 2018 vs WC 2021']['overall_agreement']); N['agr_wc25'] = pct(AGR['RF 2025 vs WC 2021']['overall_agreement'])
jb = [AGR[k]['per_class_jaccard']['Bare/sparse'] for k in AGR]
N['jac_bare_min'] = f'{min(jb):.2f}'; N['jac_bare_max'] = f'{max(jb):.2f}'
# CVA and post-processing
N['otsu'] = f'{POST["cva_otsu_threshold"]:.3f}'; N['cva_above'] = pct(POST['cva_fraction_above'])
N['ch_raw'] = f0(POST['change_ha_raw_maps']); N['ch_maj'] = f0(POST['change_ha_majority'])
N['ch_pcc'] = f0(POST['change_ha_pcc_mmu']); N['ch_cpcc'] = f0(POST['change_ha_cpcc_mmu'])
N['ch_pcc_pct'] = pct(POST['change_ha_pcc_mmu'] / tot_ha); N['ch_cpcc_pct'] = pct(POST['change_ha_cpcc_mmu'] / tot_ha)
try:
    N['ch_dw'] = f0(dw_change_area())
except Exception:  # raw Dynamic World layers not available
    N['ch_dw'] = f0(load('dw_change_area.json')['dw_pcc_change_ha'])
N['pcc_lv_bare'] = f0(TRPCC.loc['Low vegetation', 'Bare/sparse']); N['pcc_lv_built'] = f0(TRPCC.loc['Low vegetation', 'Built-up'])
N['pcc_built_lv'] = f0(TRPCC.loc['Built-up', 'Low vegetation'])
N['cva_stable_above'] = pct(CVA.loc['No change', 'share_above_otsu'])
chg = CVA.drop(index=['No change', 'Other change'])
N['cva_change_above_min'] = pct(chg.share_above_otsu.min()); N['cva_change_above_max'] = pct(chg.share_above_otsu.max())
N['cva_med_stable'] = f'{CVA.loc["No change", "median_magnitude"]:.2f}'
N['cva_med_lvbuilt'] = f'{CVA.loc["Low vegetation → Built-up", "median_magnitude"]:.2f}'
N['cva_med_lvbare'] = f'{CVA.loc["Low vegetation → Bare/sparse", "median_magnitude"]:.2f}'
# change accuracy
for m, k in (('RF PCC', 'pcc'), ('RF CPCC', 'cpcc'), ('DW PCC', 'dwp')):
    r = CH.loc[m]
    for col in ('change_UA', 'change_PA', 'builtgain_UA', 'builtgain_PA', 'change_OA', 'trajectory_OA'):
        N[f'{col.lower()}_{k}'] = pct(r[col], r[col + '_se']); N[f'{col.lower()}_{k}_v'] = pct(r[col])
    N[f'change_f1_{k}'] = f'{r.change_F1:.2f}'; N[f'builtgain_f1_{k}'] = f'{r.builtgain_F1:.2f}'
d = DIFF['RF CPCC - RF PCC']; N['diff_cpcc_pcc'] = pct(d['diff'], d['se'])
d = DIFF['RF PCC - DW PCC']; N['diff_pcc_dw'] = pct(d['diff'], d['se'])
# areas
tc = AR[AR['class'] == 'Total change'].iloc[0]; bg = AR[AR['class'] == 'Gain of built-up'].iloc[0]
N['est_change'] = ha(tc.est_ha, tc.ci95_ha); N['est_change_pct'] = pct(tc.est_ha / tot_ha)
N['stable_pct'] = f'{100 * (1 - tc.est_ha / tot_ha):.0f}'
N['overstate'] = f'{100 * (POST["change_ha_pcc_mmu"] / tc.est_ha - 1):.0f}'
N['est_builtgain'] = ha(bg.est_ha, bg.ci95_ha)
N['rel_ci_change'] = f'{100 * tc.ci95_ha / tc.est_ha:.0f}'
N['oa_ci'] = f'{100 * Z * oa("RF_2018")[1]:.1f}'; N['map_builtgain_pcc'] = f0(bg.map_ha_pcc)
t = tr('Low vegetation', 'Built-up'); N['est_lv_built'] = ha(t.est_ha, t.ci95_ha); N['n_lv_built'] = str(int(t.n_ref))
t = tr('Low vegetation', 'Bare/sparse'); N['est_lv_bare'] = ha(t.est_ha, t.ci95_ha)
N['n_ref_change'] = str(int(TR.n_ref.sum()))
e18 = ar(2018, 'Built-up'); N['map_built18'] = f0(MP.loc['Built-up', '2018_ha']); N['est_built18'] = ha(e18.est_ha, e18.ci95_ha)
e18l = ar(2018, 'Low vegetation'); N['lv_under18'] = f0(e18l.est_ha - MP.loc['Low vegetation', '2018_ha'])
s = SENS['All samples']
N['net_built'] = ha(*s['net_builtup_ha']); N['net_built_pct'] = f'{100 * s["net_builtup_ha"][0] / e18.est_ha:.0f}'
N['net_built_v'] = f0(s['net_builtup_ha'][0])
N['rel_ci_netbuilt'] = f'{100 * s["net_builtup_ha"][1] / s["net_builtup_ha"][0]:.0f}'
N['map_net_built'] = f0(MP.loc['Built-up', 'net_ha']); N['map_net_bare'] = f0(MP.loc['Bare/sparse', 'net_ha'])
# class-level net estimates from sensitivity.py
SC = json.load(open(os.path.join(T, 'sensitivity_confidence.json')))['all']
N['net_lv'] = ha(-SC['net_Low vegetation_ha'][0], SC['net_Low vegetation_ha'][1])
N['net_bare'] = ha(SC['net_Bare/sparse_ha'][0], SC['net_Bare/sparse_ha'][1])
# robustness
for key, k in (('Confidence medium or high', 'conf'), ('VHR acquired in target years', 'vhr'), ('Both restrictions', 'both'),
               ('Farther than 100 m from training', 'far')):
    s = SENS[key]; N[f'n_{k}'] = str(s['n'])
    N[f'oa18_{k}'] = f1(s['OA_2018'][0]); N[f'oa25_{k}'] = f1(s['OA_2025'][0]); N[f'ua_{k}'] = f1(s['change_UA'][0])
    N[f'tc_{k}'] = ha(*s['total_change_ha']); N[f'nb_{k}'] = ha(*s['net_builtup_ha'])
N['n_lowconf'] = str(int((SMP.confidence == 1).sum())); N['n_medconf'] = str(int((SMP.confidence == 2).sum()))
N['n_highconf'] = str(int((SMP.confidence == 3).sum()))
N['n_vhr_mis18'] = str(int((pd.to_datetime(SMP.vhr_date_2018).dt.year != 2018).sum()))
N['n_vhr_ok18'] = str(int((pd.to_datetime(SMP.vhr_date_2018).dt.year == 2018).sum()))
N['n_vhr_mis25'] = str(int((pd.to_datetime(SMP.vhr_date_2025).dt.year != 2025).sum()))
N['n_vhr_ok25'] = str(int((pd.to_datetime(SMP.vhr_date_2025).dt.year == 2025).sum()))
N['pct_vhr_mis18'] = f'{100 * (pd.to_datetime(SMP.vhr_date_2018).dt.year != 2018).mean():.0f}'
for thr in (30, 50):
    c = CLOUD.loc[thr]; N[f'oa18_c{thr}'] = f1(100 * c.OA_2018); N[f'oa25_c{thr}'] = f1(100 * c.OA_2025)
    N[f'ua_c{thr}'] = f1(100 * c.change_UA)
c = CLOUD.loc[40]; N['oa18_c40'] = f1(100 * c.OA_2018); N['oa25_c40'] = f1(100 * c.OA_2025); N['ua_c40'] = f1(100 * c.change_UA)
N['sep_med'] = f0(SEP['median_m']); N['sep_30'] = str(SEP['n_within_30m']); N['sep_100'] = str(SEP['n_within_100m'])
N['sep_same'] = str(SEP['n_identical'])
# experiments
ex = EXP.set_index(['feature_set', 'classifier'])
g = lambda fs, c, y: 100 * ex.loc[(fs, c), f'OA_{y}']
N['rf_sb18'] = f1(g('S-B', 'RF', 2018)); N['rf_ssb18'] = f1(g('SS-B', 'RF', 2018))
N['rf_sb25'] = f1(g('S-B', 'RF', 2025)); N['rf_ssb25'] = f1(g('SS-B', 'RF', 2025))
P = PAIR
N['d_spring18'] = f'{P["RF SS-B - RF S-B"]["2018"]["diff_pp"]:.1f} ± {P["RF SS-B - RF S-B"]["2018"]["ci95_pp"]:.1f}'
N['d_spring25'] = f'{P["RF SS-B - RF S-B"]["2025"]["diff_pp"]:.1f} ± {P["RF SS-B - RF S-B"]["2025"]["ci95_pp"]:.1f}'
dI = [abs(g('S-BI', 'RF', y) - g('S-B', 'RF', y)) for y in (2018, 2025)] + [abs(g('SS-BI', 'RF', y) - g('SS-B', 'RF', y)) for y in (2018, 2025)]
N['d_idx_max'] = f'{max(dI):.1f}'
dT = [abs(g('SS-BI-T', 'RF', y) - g('SS-BI', 'RF', y)) for y in (2018, 2025)]; N['d_tex_max'] = f'{max(dT):.1f}'
s18 = [g(fs, 'SVM', 2018) - g(fs, 'RF', 2018) for fs in ex.index.levels[0]]
s25 = [g(fs, 'SVM', 2025) - g(fs, 'RF', 2025) for fs in ex.index.levels[0]]
N['svm18_min'] = f1(min(s18)); N['svm18_max'] = f1(max(s18)); N['svm25_min'] = f1(min(s25)); N['svm25_max'] = f1(max(s25))
N['svm_best18'] = f1(g('SS-BI', 'SVM', 2018)); N['svm_best25'] = f1(g('SS-B', 'SVM', 2025))
N['chf1_min'] = f'{EXP.change_F1.min():.2f}'; N['chf1_max'] = f'{EXP.change_F1.max():.2f}'
N['chua_max'] = f0(100 * EXP.change_UA.max())
for gname, k in (('Spring bands', 'spb'), ('Summer bands', 'smb'), ('Spring indices', 'spi'), ('Summer indices', 'smi'), ('Texture', 'tex')):
    N[f'perm_{k}'] = f1(PERM.loc[gname, 'mean_drop_pp'])
N['mdi_top'] = IMP.iloc[0, 0]

if __name__ == '__main__':
    for k, v in N.items():
        print(k, '=', v)
