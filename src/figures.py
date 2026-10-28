# -*- coding: utf-8 -*-
"""Revised publication figure set (300 dpi PNG): main figures fig01-fig10 and supplementary figS1-figS2."""
import json, io, math, os, time
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Rectangle, Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image
from pyproj import Transformer
from common import *
from estimators import Z95
from change import cva_magnitude
import print_size
print_size.enable()  # minimum 8 pt at printed size
if os.environ.get('FIG_LANG') == 'tr':
    import fig_tr  # noqa: F401  (Turkish labels)
    FIGS = os.path.join(ROOT, 'outputs', 'figures_tr'); os.makedirs(FIGS, exist_ok=True)

FW, FT = 6.0, 8.5     # figure width (in) = printed width 15.2 cm; panel title size (pt)
plt.rcParams.update({'font.family': 'Palatino Linotype', 'font.size': 8, 'axes.linewidth': 0.6,
                     'axes.titlesize': FT, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'legend.title_fontsize': 8.5,
                     'axes.edgecolor': '#444444', 'xtick.color': '#333333', 'ytick.color': '#333333',
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03})
BIG = {'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
       'legend.fontsize': 9}   # map and accuracy figures with dense labels: 9 pt floor at printed size
INK, GRID = '#222222', '#dddddd'
y1, y2 = YEARS
NAMES = [CLASSES[c] for c in CLASSES]
CM = ListedColormap(['#ffffff'] + [COLORS[c] for c in CLASSES])
prof = read(f'{MAPS}/lulc_{y1}_rf.tif')[1]
T = prof['transform']; H, W = prof['height'], prof['width']
EXT = (T.c, T.c + W * T.a, T.f + H * T.e, T.f)
to_utm = Transformer.from_crs('EPSG:4326', CFG['project']['crs'], always_xy=True)
to_ll = Transformer.from_crs(CFG['project']['crs'], 'EPSG:4326', always_xy=True)
CHG = {1: ('Change to built-up', '#7a0177'), 2: ('Change to bare/sparse', '#8c510a'),
       3: ('Change to low vegetation', '#41ab5d'), 4: ('Other change', '#525252')}
STRC = {1: '#9ecae1', 2: '#6d9d78', 3: '#eef5d0', 4: '#e9b3cf', 5: '#f9dcaa', 6: '#7a0177', 7: '#8c510a', 8: '#737373'}
STRN = {1: 'Stable water', 2: 'Stable tree cover', 3: 'Stable low vegetation', 4: 'Stable built-up',
        5: 'Stable bare/sparse', 6: 'CPCC change to built-up', 7: 'CPCC other change', 8: 'PCC change without CVA support'}
sp = pd.read_csv(f'{TABS}/S_reference_sample.csv')
sp = sp.merge(pd.read_csv(f'{SAMP}/validation_points.csv')[['pid', 'row', 'col']], on='pid')
tp = pd.read_csv(f'{SAMP}/training_points.csv')


def rgb(year, season='summer', bands=('B4', 'B3', 'B2'), gain=3.2):
    a, _, n = read(f'{PROC}/S2_{year}_features.tif')
    x = np.stack([a[n.index(f'{b}_{season}')] for b in bands], -1).astype(np.float32) / 10000
    return np.clip(x * gain, 0, 1) ** (1 / 1.4)


def polys(g):
    if g['type'] == 'Polygon':
        return [g['coordinates']]
    if g['type'] == 'MultiPolygon':
        return g['coordinates']
    if g['type'] == 'GeometryCollection':
        return [p for s in g['geometries'] for p in polys(s)]
    return []


def draw_geojson(ax, path, proj=None, **kw):
    g = json.load(open(os.path.join(ROOT, 'data', 'raw', path)))
    for f in g['features']:
        for p in polys(f['geometry']):
            xy = np.array(p[0])
            if proj:
                xy = np.array([proj.transform(*c) for c in xy])
            ax.plot(xy[:, 0], xy[:, 1], **kw)
    return g


def scalebar(ax, ext, length, label, loc=(0.05, 0.06), color='black'):
    x0, x1, yb, yt = ext
    sx, sy = x0 + loc[0] * (x1 - x0), yb + loc[1] * (yt - yb); h = 0.014 * (yt - yb)
    ax.add_patch(Rectangle((sx, sy), length / 2, h, fc='black', ec='black', lw=0.5, zorder=5))
    ax.add_patch(Rectangle((sx + length / 2, sy), length / 2, h, fc='white', ec='black', lw=0.5, zorder=5))
    ax.text(sx + length / 2, sy + 2.2 * h, label, ha='center', va='bottom', zorder=5,
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.8))


def north(ax, ext, pos=(0.95, 0.80)):
    x0, x1, yb, yt = ext
    x, y = x0 + pos[0] * (x1 - x0), yb + pos[1] * (yt - yb)
    ax.annotate('N', xy=(x, y + 0.12 * (yt - yb)), xytext=(x, y), ha='center', va='top', fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1), zorder=6,
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=0.5))


def km_ticks(ax, ext=EXT, step=5000):
    x0, x1, yb, yt = ext
    xt = np.arange(np.ceil(x0 / step) * step, x1, step); yt_ = np.arange(np.ceil(yb / step) * step, yt, step)
    ax.set_xticks(xt); ax.set_yticks(yt_)
    ax.set_xticklabels([f'{v / 1000:.0f}' for v in xt]); ax.set_yticklabels([f'{v / 1000:.0f}' for v in yt_])
    ax.tick_params(length=2, pad=1)


def noticks(ax):
    ax.set_xticks([]); ax.set_yticks([])


def change_code(l1, l2):
    c = np.zeros(l1.shape, np.uint8)
    ch = (l1 != l2) & (l1 > 0) & (l2 > 0)
    c[ch & (l2 == 4)] = 1; c[ch & (l2 == 5)] = 2; c[ch & (l2 == 3)] = 3
    c[ch & (c == 0)] = 4
    return c


CCM = ListedColormap(['#f4f4f4'] + [CHG[k][1] for k in CHG])


# ---------------------------------------------------------------- Figure 1 -----------------
def fig01():
    fig = plt.figure(figsize=(FW, 3.35))
    a = fig.add_axes([0.0, 0.55, 0.31, 0.40]); b = fig.add_axes([0.0, 0.02, 0.31, 0.45]); c = fig.add_axes([0.40, 0.09, 0.60, 0.84])
    g = json.load(open(os.path.join(ROOT, 'data', 'raw', 'turkey_provinces.geojson')))
    for f in g['features']:
        for p in polys(f['geometry']):
            xy = np.array(p[0]); ank = f['properties']['ADM1_NAME'] == 'Ankara'
            a.fill(xy[:, 0], xy[:, 1], fc='#9ecae1' if ank else '#eeeeee', ec='#9a9a9a', lw=0.25)
    a.set_xlim(25.6, 44.9); a.set_ylim(35.7, 42.3); a.set_aspect(1.3)
    a.set_xticks([28, 34, 40]); a.set_yticks([37, 40]); a.tick_params(length=2, pad=1)
    a.set_xticklabels([f'{v}°E' for v in [28, 34, 40]]); a.set_yticklabels([f'{v}°N' for v in [37, 40]])
    a.set_title('(a) Türkiye', fontsize=FT, fontweight='bold', loc='left', pad=2)
    lon0, lat0, lon1, lat1 = CFG['study_area']['bbox_wgs84']
    a.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fc='none', ec='red', lw=1.0))
    ea = (25.6, 44.9, 35.7, 42.3)
    scalebar(a, ea, 300 / (111.32 * np.cos(np.radians(39))), '300 km', loc=(0.05, 0.08)); north(a, ea, pos=(0.94, 0.62))
    gg = draw_geojson(b, 'ankara_all_districts.geojson', color='#8a8a8a', lw=0.35)
    for f in gg['features']:
        for p in polys(f['geometry']):
            xy = np.array(p[0]); b.fill(xy[:, 0], xy[:, 1], fc='#deebf7', ec='none', zorder=0)
    b.add_patch(Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fc='none', ec='red', lw=1.1))
    b.plot(32.854, 39.920, marker='*', ms=6, color='black'); b.text(32.97, 39.86, 'Ankara\ncentre', va='top', linespacing=1.1)
    b.set_aspect(1.3); b.tick_params(length=2, pad=1)
    eb = b.get_xlim() + b.get_ylim()
    scalebar(b, eb, 50 / (111.32 * np.cos(np.radians(39.5))), '50 km', loc=(0.06, 0.06)); north(b, eb, pos=(0.92, 0.80))
    b.set_xticks([31.5, 32.5, 33.5]); b.set_xticklabels(['31.5°E', '32.5°E', '33.5°E'])
    b.set_yticks([39, 40]); b.set_yticklabels(['39°N', '40°N'])
    b.set_title('(b) Ankara province', fontsize=FT, fontweight='bold', loc='left', pad=2)
    c.imshow(rgb(y2), extent=EXT, interpolation='bilinear')
    draw_geojson(c, 'ankara_districts.geojson', proj=to_utm, color='white', lw=0.8, alpha=0.9)
    for name, lo, la in (('Sincan', 32.575, 39.968), ('Etimesgut', 32.672, 39.948), ('Yenimahalle', 32.735, 39.992),
                         ('Çankaya', 32.735, 39.885)):
        x, y = to_utm.transform(lo, la)
        c.text(x, y, name, color='white', ha='center', fontweight='bold', bbox=dict(fc='black', ec='none', alpha=0.4, pad=1))
    c.set_xlim(EXT[0], EXT[1]); c.set_ylim(EXT[2], EXT[3]); km_ticks(c)
    scalebar(c, EXT, 5000, '5 km'); north(c, EXT, pos=(0.05, 0.80))
    c.set_xlabel('Easting (km)'); c.set_ylabel('Northing (km)', labelpad=2)
    c.set_title('(c) Study area, Sentinel-2 summer 2025 composite', fontsize=FT, fontweight='bold', loc='left', pad=3)
    c.text(0.995, 0.012, 'WGS 84 / UTM zone 36N (EPSG:32636)', transform=c.transAxes, ha='right', color='white',
           bbox=dict(fc='black', ec='none', alpha=0.45, pad=1))
    fig.savefig(f'{FIGS}/fig01_study_area.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 2 -----------------
def fig02():
    fig, ax = plt.subplots(figsize=(FW, 4.5)); fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(-1.5, 101.5); ax.set_ylim(2, 80); ax.axis('off')
    def box(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec='#555555', lw=0.7))
        ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', color=INK, linespacing=1.25)
    def arr(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='#555555', lw=0.8))
    c1, c2, c3 = '#e8f1fa', '#eef6e6', '#fbeee6'
    X, BW = (0, 34.5, 69), 31
    box(X[0], 64, BW, 14, 'Sentinel-2 L2A surface\nreflectance, 2018 and 2025\nSCL + s2cloudless masking', c1)
    box(X[1], 64, BW, 14, 'Spring and summer\nmedian composites\n10 bands, 6 indices,\n3 texture layers', c1)
    box(X[2], 64, BW, 14, 'Dynamic World (2018, 2025)\nWorldCover 2021\nGHSL built-up 2018', c1)
    box(X[2], 43, BW, 14, 'Stable consensus\ntraining pixels (agreement,\nprobability, homogeneity,\nGHSL screening)', c2)
    box(X[1], 43, BW, 14, 'Random Forest trained\non pooled samples and\napplied to both years\n3 × 3 majority filter', c2)
    box(X[0], 43, BW, 14, 'PCC and CVA-constrained\nPCC (Otsu threshold)\nMMU = 5 pixels\ntransition matrices', c2)
    box(X[0], 22, BW, 14, 'Stratified random sample\n(8 strata, n = 500)\nblind VHR interpretation\nfor both dates', c3)
    box(X[1], 22, BW, 14, 'Design-based estimators:\nOA, UA, PA, area-proportion\nerror matrices, areas\nwith 95% CI', c3)
    box(X[2], 22, BW, 14, 'Comparisons and robustness:\nRF vs SVM, feature sets,\nDynamic World PCC, cloud\nthreshold, reference subsets', c3)
    box(17, 4, 66, 9, 'Area-adjusted estimates of land cover change\nand transitions with 95% confidence intervals', '#ffffff')
    m = [x + BW / 2 for x in X]
    arr(31.4, 71, 34.1, 71); arr(m[1], 63.6, m[1], 57.4); arr(m[2], 63.6, m[2], 57.4)
    arr(68.6, 50, 65.9, 50); arr(34.1, 50, 31.4, 50); arr(m[0], 42.6, m[0], 36.4)
    arr(31.4, 29, 34.1, 29); arr(65.9, 29, 68.6, 29); arr(m[1], 21.6, m[1], 13.4)
    fig.savefig(f'{FIGS}/fig02_workflow.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 3 -----------------
def fig03():
    s = read(f'{MAPS}/strata.tif')[0][0]
    fig = plt.figure(figsize=(FW, 4.6))
    a = fig.add_axes([0.0, 0.53, 0.49, 0.44], anchor='S'); b = fig.add_axes([0.51, 0.53, 0.49, 0.44], anchor='S')
    c = fig.add_axes([0.10, 0.09, 0.34, 0.35])
    a.imshow(s, cmap=ListedColormap(['#ffffff'] + [STRC[h] for h in STRN]), vmin=0, vmax=8, extent=EXT, interpolation='nearest')
    for h in STRN:
        d = sp[sp.stratum == h]
        a.scatter(d.x_utm36n, d.y_utm36n, s=5, c='black', marker='o', lw=0, zorder=4)
    a.set_title('(a) Sampling strata and validation pixels (n = 500)', loc='left', pad=3); noticks(a)
    scalebar(a, EXT, 5000, '5 km'); north(a, EXT)
    gray = rgb(y2).mean(-1)
    b.imshow(gray, cmap='Greys_r', extent=EXT, vmin=0.1, vmax=1.0, interpolation='bilinear')
    for cl in CLASSES:
        d = tp[tp.class_id == cl]
        b.scatter(d.x, d.y, s=3.5, c=COLORS[cl], lw=0.15, edgecolors='black', zorder=4)
    b.set_title('(b) Training locations by class', loc='left', pad=3); noticks(b)
    h1 = [Patch(fc=STRC[h], ec='#555555', lw=0.3, label=STRN[h]) for h in STRN]
    fig.legend(handles=h1 + [Line2D([], [], marker='o', ls='', color='black', ms=3, label='Validation pixel')],
               loc='upper left', bbox_to_anchor=(0.49, 0.50), ncol=1, frameon=False, title='Strata', alignment='left',
               handlelength=1.4, handletextpad=0.5, labelspacing=0.35)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc=COLORS[k], mec='black', mew=0.3, ms=5, label=CLASSES[k]) for k in CLASSES],
               loc='upper left', bbox_to_anchor=(0.815, 0.50), ncol=1, frameon=False, title='Training class', alignment='left',
               handlelength=1.0, handletextpad=0.4, labelspacing=0.35)
    c.hist(sp.dist_training_m, bins=np.arange(0, 1300, 50), color='#3a7dc9', ec='white', lw=0.4)
    c.axvline(np.median(sp.dist_training_m), color=INK, lw=0.8, ls='--')
    c.text(0.98, 0.95, f'median {np.median(sp.dist_training_m):.0f} m', transform=c.transAxes, ha='right', va='top')
    c.set_xlabel('Distance to nearest training location (m)'); c.set_ylabel('Validation pixels')
    c.set_xticks([0, 400, 800, 1200])
    c.set_title('(c) Spatial separation', loc='left', pad=3)
    for sname in ('top', 'right'):
        c.spines[sname].set_visible(False)
    fig.savefig(f'{FIGS}/fig03_sampling_design.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 4 -----------------
def fig04():
    l1 = read(f'{MAPS}/lulc_{y1}_rf.tif')[0][0]; l2 = read(f'{MAPS}/lulc_{y2}_pcc.tif')[0][0]; l3 = read(f'{MAPS}/lulc_{y2}_cpcc.tif')[0][0]
    fig = plt.figure(figsize=(FW, 5.3))
    axs = np.array([[fig.add_axes([0.0, 0.60, 0.49, 0.37]), fig.add_axes([0.51, 0.60, 0.49, 0.37])],
                    [fig.add_axes([0.0, 0.14, 0.49, 0.37]), fig.add_axes([0.51, 0.14, 0.49, 0.37])]])
    for ax, im, t in ((axs[0, 0], l1, f'(a) RF land cover {y1}'), (axs[0, 1], l2, f'(b) RF land cover {y2}')):
        ax.imshow(im, cmap=CM, vmin=0, vmax=5, extent=EXT, interpolation='nearest'); ax.set_title(t, loc='left', pad=3); noticks(ax)
    for ax, im, t in ((axs[1, 0], change_code(l1, l2), '(c) PCC change'), (axs[1, 1], change_code(l1, l3), '(d) CVA-constrained PCC change')):
        ax.imshow(im, cmap=CCM, vmin=0, vmax=4, extent=EXT, interpolation='nearest'); ax.set_title(t, loc='left', pad=3); noticks(ax)
    scalebar(axs[0, 0], EXT, 5000, '5 km'); north(axs[0, 0], EXT)
    h1 = [Patch(fc=COLORS[c], ec='#555555', lw=0.3, label=CLASSES[c]) for c in CLASSES]
    h2 = [Patch(fc='#f4f4f4', ec='#555555', lw=0.3, label='No change')] + [Patch(fc=CHG[k][1], label=CHG[k][0]) for k in CHG]
    fig.legend(handles=h1, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.60), columnspacing=1.2, handlelength=1.4)
    fig.legend(handles=h2, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.135), columnspacing=1.2, handlelength=1.4)
    fig.savefig(f'{FIGS}/fig04_maps_change.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 5 -----------------
def fig05():
    M = read(f'{MAPS}/cva_magnitude.tif')[0][0] / 1000
    post = json.load(open(f'{TABS}/change_area_postprocessing.json')); thr = post['cva_otsu_threshold']
    fig = plt.figure(figsize=(FW, 4.8))
    a = fig.add_axes([0.0, 0.50, 0.55, 0.47]); b = fig.add_axes([0.745, 0.585, 0.25, 0.36]); c = fig.add_axes([0.09, 0.15, 0.90, 0.26])
    im = a.imshow(np.clip(M, 0, 0.8), cmap='Greys', vmin=0, vmax=0.8, extent=EXT, interpolation='bilinear')
    noticks(a); a.set_title('(a) CVA magnitude', loc='left', pad=3); scalebar(a, EXT, 5000, '5 km')
    cax = fig.add_axes([0.565, 0.56, 0.015, 0.36]); fig.colorbar(im, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8])
    Mf, valid = cva_magnitude()   # same 500,000 pixels as the Otsu threshold (change.py)
    v = Mf.ravel()[np.random.default_rng(SEED).choice(np.flatnonzero(valid), 500000, replace=False)]
    b.hist(v, bins=np.linspace(0, 1.2, 97), color='#9a9a9a', ec='none')
    b.hist(v[v > thr], bins=np.linspace(0, 1.2, 97), color='#c51b7d', ec='none')
    b.axvline(thr, color=INK, lw=0.9, ls='--')
    b.text(thr + 0.04, b.get_ylim()[1] * 0.97, f'Otsu threshold = {thr:.3f}\n{100 * post['cva_fraction_above']:.1f}% of pixels above', va='top')
    b.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x / 1000:.0f}'))
    b.set_xticks([0, 0.4, 0.8, 1.2])
    b.set_xlabel('Magnitude (reflectance units)'); b.set_ylabel('Pixels (10³)')
    b.set_title('(b) Magnitude histogram', loc='left', pad=3)
    order = ['No change', 'Low vegetation → Built-up', 'Low vegetation → Bare/sparse', 'Bare/sparse → Low vegetation',
             'Bare/sparse → Built-up', 'Other change']
    data = [sp.loc[sp.ref_transition == o, 'cva'].values if 'ref_transition' in sp else None for o in order]
    bp = c.boxplot(data, orientation='vertical', widths=0.5, patch_artist=True, showfliers=True,
                   flierprops=dict(marker='o', ms=2, mfc='#777777', mec='none'), medianprops=dict(color=INK, lw=1))
    for patch, col in zip(bp['boxes'], ['#bdbdbd', '#c51b7d', '#f1a340', '#d9ef8b', '#7a0177', '#737373']):
        patch.set_facecolor(col); patch.set_edgecolor('#444444'); patch.set_linewidth(0.6)
    c.axhline(thr, color=INK, lw=0.9, ls='--')
    lab = [o.replace('Low vegetation', 'Low veg.').replace(' → ', ' to\n') for o in order]
    c.set_xticks(range(1, 7)); c.set_xticklabels([f'{o}\n(n = {len(d)})' for o, d in zip(lab, data)], linespacing=1.15)
    c.set_ylabel('Magnitude'); c.set_ylim(0, 1.2); c.set_yticks([0, 0.4, 0.8, 1.2])
    c.set_title('(c) Magnitude at validation pixels by reference transition', loc='left', pad=3)
    for ax in (b, c):
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.savefig(f'{FIGS}/fig05_cva.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 6 -----------------
HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
       'Referer': 'https://livingatlas.arcgis.com/wayback/'}


def wb_tile(rid, z, x, y):
    p = os.path.join(ROOT, 'data', 'reference', 'wayback_tiles', str(rid), f'{z}_{x}_{y}.jpg')
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        for _ in range(3):
            try:
                r = requests.get(f'https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/'
                                 f'MapServer/tile/{rid}/{z}/{y}/{x}', headers=HDR, timeout=30)
                if r.status_code == 200:
                    open(p, 'wb').write(r.content); break
            except Exception:
                time.sleep(2)
    try:
        return Image.open(p).convert('RGB')
    except Exception:
        return Image.new('RGB', (256, 256), (90, 90, 90))


def wb_window(rid, x0, y0, x1, y1, z=16):
    n = 256 * 2 ** z
    def mp(lon, lat):
        return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    (lo0, la1), (lo1, la0) = to_ll.transform(x0, y1), to_ll.transform(x1, y0)
    (px0, py0), (px1, py1) = mp(lo0, la1), mp(lo1, la0)
    tx0, ty0, tx1, ty1 = int(px0 // 256), int(py0 // 256), int(px1 // 256), int(py1 // 256)
    mos = Image.new('RGB', ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mos.paste(wb_tile(rid, z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    return mos.crop((int(px0 - tx0 * 256), int(py0 - ty0 * 256), int(px1 - tx0 * 256), int(py1 - ty0 * 256)))


SITES = [('Yapracık and Bağlıca', 32.585, 39.905), ('Etimesgut and Eryaman', 32.655, 39.975), ('Northern Sincan', 32.555, 39.985)]


def fig06():
    l1 = read(f'{MAPS}/lulc_{y1}_rf.tif')[0][0]; l2 = read(f'{MAPS}/lulc_{y2}_pcc.tif')[0][0]
    cc = change_code(l1, l2)
    fig, axs = plt.subplots(3, 5, figsize=(FW, 4.85), gridspec_kw={'wspace': 0.03, 'hspace': 0.05})
    fig.subplots_adjust(left=0.07, right=1.0, top=0.95, bottom=0.22)
    half = 150
    for i, (name, lo, la) in enumerate(SITES):
        x, y = to_utm.transform(lo, la); c = int((x - T.c) / T.a); r = int((y - T.f) / T.e)
        r0, c0 = max(r - half, 0), max(c - half, 0)
        sl = (slice(r0, r0 + 2 * half), slice(c0, c0 + 2 * half))
        X0, Y1 = T.c + c0 * T.a, T.f + r0 * T.e; X1, Y0 = X0 + 2 * half * 10, Y1 - 2 * half * 10
        ext = (X0, X1, Y0, Y1)
        for j in range(5):
            ax = axs[i, j]
            if j < 2:
                ax.imshow(wb_window((4756, 26334)[j], X0, Y0, X1, Y1), extent=ext, interpolation='bilinear')
            elif j < 4:
                ax.imshow((l1, l2)[j - 2][sl], cmap=CM, vmin=0, vmax=5, extent=ext, interpolation='nearest')
            else:
                ax.imshow(cc[sl], cmap=CCM, vmin=0, vmax=4, extent=ext, interpolation='nearest')
                d = sp[(sp.x_utm36n > X0) & (sp.x_utm36n < X1) & (sp.y_utm36n > Y0) & (sp.y_utm36n < Y1)]
                rc = d.ref_2018 != d.ref_2025
                ax.scatter(d.x_utm36n[rc], d.y_utm36n[rc], marker='^', s=18, c='white', edgecolors='black', lw=0.6, zorder=5)
                ax.scatter(d.x_utm36n[~rc], d.y_utm36n[~rc], marker='o', s=13, c='black', edgecolors='white', lw=0.5, zorder=5)
            ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); noticks(ax)
            if i == 0:
                ax.set_title((f'VHR {y1}', f'VHR {y2}', f'RF {y1}', f'RF {y2}', 'PCC change')[j], pad=3)
        axs[i, 0].set_ylabel(f'({"abc"[i]}) {name}'.replace(' and ', ' and\n'), linespacing=1.1, labelpad=3)
        scalebar(axs[i, 0], ext, 1000, '1 km', loc=(0.08, 0.06))
    h = [Patch(fc=COLORS[k], ec='#555555', lw=0.3, label=CLASSES[k]) for k in CLASSES] + \
        [Patch(fc=CHG[k][1], label=CHG[k][0]) for k in CHG] + \
        [Line2D([], [], marker='^', ls='', mfc='white', mec='black', ms=5.5, label='Reference: change'),
         Line2D([], [], marker='o', ls='', mfc='black', mec='white', ms=5, label='Reference: no change')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.0), columnspacing=1.2,
               handlelength=1.4, labelspacing=0.3)
    fig.savefig(f'{FIGS}/fig06_examples.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 7 -----------------
def fig07():
    acc = pd.read_csv(f'{TABS}/accuracy_by_class.csv'); ch = pd.read_csv(f'{TABS}/change_accuracy.csv').set_index('method')
    maps = [('RF_2018', f'RF {y1}', '#3a7dc9', 'o'), ('RF_PCC_2025', f'RF {y2}', '#3a7dc9', 's'),
            ('DW_2018', f'DW {y1}', '#e08214', 'o'), ('DW_2025', f'DW {y2}', '#e08214', 's')]
    fig = plt.figure(figsize=(FW, 3.7))
    axs = [fig.add_axes([0.075, 0.33, 0.25, 0.60]), fig.add_axes([0.375, 0.33, 0.25, 0.60]), fig.add_axes([0.835, 0.33, 0.16, 0.60])]
    xb = np.arange(5)
    for ax, met, t in ((axs[0], 'UA', "(a) User's accuracy"), (axs[1], 'PA', "(b) Producer's accuracy")):
        for k, (m, lab, col, mk) in enumerate(maps):
            d = acc[acc['map'] == m].set_index('class_name').loc[NAMES]
            ax.errorbar(xb + (k - 1.5) * 0.17, 100 * d[met], yerr=100 * Z95 * d[f'{met}_se'], fmt=mk, ms=3.8, color=col,
                        mfc=col if mk == 'o' else 'white', capsize=1.5, lw=0.8, label=lab)
        ax.set_xticks(xb); ax.set_xticklabels(['Water', 'Tree', 'Low veg.', 'Built-up', 'Bare'], rotation=40, ha='right',
                                              rotation_mode='anchor')
        ax.set_ylim(0, 105); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title(t, loc='left', pad=3)
    axs[0].set_ylabel('Accuracy (%)')
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc='lower center', bbox_to_anchor=(0.36, 0.0), ncol=4,
               columnspacing=1.0, handletextpad=0.3)
    ax = axs[2]
    mets = [('change_UA', 'Change UA'), ('change_PA', 'Change PA'), ('builtgain_UA', 'Built-up gain UA'),
            ('builtgain_PA', 'Built-up gain PA'), ('change_OA', 'Binary OA'), ('trajectory_OA', 'Trajectory OA')]
    yb = np.arange(len(mets))[::-1]
    for k, (m, col, mk) in enumerate((('RF PCC', '#3a7dc9', 'o'), ('RF CPCC', '#7a0177', 'D'), ('DW PCC', '#e08214', 's'))):
        vals = [100 * ch.loc[m, a] for a, _ in mets]; err = [100 * Z95 * ch.loc[m, a + '_se'] for a, _ in mets]
        ax.errorbar(vals, yb + (1 - k) * 0.22, xerr=err, fmt=mk, ms=3.8, color=col, capsize=1.5, lw=0.8, label=m)
    ax.set_yticks(yb); ax.set_yticklabels([l for _, l in mets]); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_xlabel('Accuracy (%)')
    ax.set_title('(c) Change products', loc='left', pad=3)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', bbox_to_anchor=(0.91, -0.02), ncol=1,
               handletextpad=0.3, labelspacing=0.25)
    for a_ in axs:
        for s in ('top', 'right'):
            a_.spines[s].set_visible(False)
    fig.savefig(f'{FIGS}/fig07_accuracy.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 8 -----------------
def fig08():
    mats = [pd.read_csv(f'{TABS}/transition_matrix_pcc_ha.csv', index_col=0), pd.read_csv(f'{TABS}/transition_matrix_cpcc_ha.csv', index_col=0),
            pd.read_csv(f'{TABS}/transition_reference_estimated_ha.csv', index_col=0)]
    titles = ['(a) PCC map', '(b) CPCC map', '(c) Reference estimate']
    fig = plt.figure(figsize=(FW, 5.6))
    axs = [fig.add_axes([0.14, 0.585, 0.38, 0.37]), fig.add_axes([0.60, 0.585, 0.38, 0.37]), fig.add_axes([0.14, 0.075, 0.38, 0.37])]
    norm = LogNorm(vmin=10, vmax=4000); cmap = plt.get_cmap('Purples')
    ab = ['Water', 'Tree', 'Low veg.', 'Built-up', 'Bare']
    for ax, M, t in zip(axs, mats, titles):
        v = M.values
        off = np.where(np.eye(5, dtype=bool), np.nan, np.maximum(v, 1e-3))
        ax.imshow(off, cmap=cmap, norm=norm, aspect='auto')
        for i in range(5):
            for j in range(5):
                val = v[i, j]
                if i == j:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc='#eeeeee', ec='white', lw=1))
                    ax.text(j, i, f'{val:,.0f}', ha='center', va='center', color='#555555')
                else:
                    dark = val > 600
                    ax.text(j, i, f'{val:,.0f}', ha='center', va='center', color='white' if dark else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels(ab, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_yticks(range(5)); ax.set_yticklabels(ab if ax is not axs[1] else [])
        ax.set_title(t, loc='left', pad=3); ax.set_xlabel(f'Class in {y2}')
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel(f'Class in {y1}'); axs[2].set_ylabel(f'Class in {y1}')
    cax = fig.add_axes([0.60, 0.075, 0.022, 0.37])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, cax=cax)
    cb.set_label('Transition area (ha)')
    fig.savefig(f'{FIGS}/fig08_transitions.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 9 -----------------
def fig09():
    a = pd.read_csv(f'{TABS}/area_estimates.csv'); mp = pd.read_csv(f'{TABS}/class_area_map_pcc.csv', index_col=0)
    sens = json.load(open(f'{TABS}/sensitivity_confidence.json'))['all']
    fig = plt.figure(figsize=(FW, 3.4))
    axs = [fig.add_axes([0.215, 0.27, 0.33, 0.66]), fig.add_axes([0.79, 0.27, 0.20, 0.66])]
    ax = axs[0]; rows = []
    for c in NAMES:
        for yr in (y1, y2):
            e = a[(a.year.astype(str) == str(yr)) & (a['class'] == c)].iloc[0]
            rows.append((f'{c} {yr}', mp.loc[c, f'{yr}_ha'], e.est_ha, e.ci95_ha, [k for k in CLASSES if CLASSES[k] == c][0]))
    yb = np.arange(len(rows))[::-1]
    for yy, (lab, m, e, ci, k) in zip(yb, rows):
        ax.plot([m / 1000, e / 1000], [yy, yy], color='#bbbbbb', lw=1.2, zorder=1)
        ax.errorbar(e / 1000, yy, xerr=ci / 1000, fmt='o', ms=5, color=COLORS[k], mec='#333333', mew=0.4, capsize=2, lw=1, zorder=3)
        ax.scatter(m / 1000, yy, marker='x', s=22, color=INK, lw=1, zorder=4)
    ax.set_yticks(yb); ax.set_yticklabels([r[0] for r in rows]); ax.set_xticks([0, 10, 20])
    ax.set_xlabel('Area (10³ ha)'); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title('(a) Class area: map count vs estimate', loc='left', pad=3)
    fig.legend(handles=[Line2D([], [], marker='x', ls='', color=INK, label='Map pixel count'),
                        Line2D([], [], marker='o', ls='-', color='#777777', mec='#333333', label='Estimate ± 95% CI')],
               frameon=False, loc='lower center', bbox_to_anchor=(0.38, 0.02), ncol=2, handletextpad=0.4)
    ax = axs[1]
    tc = a[a['class'] == 'Total change'].iloc[0]; bg = a[a['class'] == 'Gain of built-up'].iloc[0]
    q = [('Total change', tc.map_ha_pcc, tc.map_ha, tc.est_ha, tc.ci95_ha),
         ('Built-up gain', bg.map_ha_pcc, bg.map_ha, bg.est_ha, bg.ci95_ha),
         ('Net built-up', mp.loc['Built-up', 'net_ha'], None, *sens['net_Built-up_ha']),
         ('Net low vegetation', mp.loc['Low vegetation', 'net_ha'], None, *sens['net_Low vegetation_ha']),
         ('Net bare/sparse', mp.loc['Bare/sparse', 'net_ha'], None, *sens['net_Bare/sparse_ha'])]
    yb = np.arange(len(q))[::-1]
    for yy, (lab, m1, m2, e, ci) in zip(yb, q):
        ax.errorbar(e / 1000, yy, xerr=ci / 1000, fmt='o', ms=5, color='#3a7dc9', mec='#333333', mew=0.4, capsize=2, lw=1, zorder=3)
        ax.scatter(m1 / 1000, yy + 0.15, marker='x', s=22, color=INK, lw=1, zorder=4)
        if m2 is not None:
            ax.scatter(m2 / 1000, yy - 0.15, marker='+', s=30, color='#7a0177', lw=1.1, zorder=4)
    ax.axvline(0, color='#888888', lw=0.6)
    ax.set_yticks(yb); ax.set_yticklabels([r[0] for r in q]); ax.set_xlabel('Area (10³ ha)'); ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(-0.6, len(q) - 0.4)
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title(f'(b) Change {y1}–{y2}', loc='left', pad=3)
    fig.legend(handles=[Line2D([], [], marker='x', ls='', color=INK, label='PCC map'),
                        Line2D([], [], marker='+', ls='', color='#7a0177', label='CPCC map'),
                        Line2D([], [], marker='o', ls='-', color='#3a7dc9', label='Estimate ± 95% CI')],
               frameon=False, loc='lower center', bbox_to_anchor=(0.86, -0.03), ncol=1, handletextpad=0.4, labelspacing=0.2)
    for a_ in axs:
        for s in ('top', 'right'):
            a_.spines[s].set_visible(False)
    fig.savefig(f'{FIGS}/fig09_area_estimates.png'); plt.close(fig)


# ---------------------------------------------------------------- Figure 10 ----------------
def fig10():
    e = pd.read_csv(f'{TABS}/experiments_classifier_featureset.csv'); pi = pd.read_csv(f'{TABS}/permutation_importance_grouped.csv')
    fig = plt.figure(figsize=(FW, 3.5))
    axs = [fig.add_axes([0.09, 0.30, 0.44, 0.63]), fig.add_axes([0.765, 0.30, 0.225, 0.63])]
    ax = axs[0]; fs = list(dict.fromkeys(e.feature_set)); xb = np.arange(len(fs))
    for k, (clf, col) in enumerate((('RF', '#3a7dc9'), ('SVM', '#c51b7d'))):
        for j, (yr, mk) in enumerate(((y1, 'o'), (y2, 's'))):
            d = e[e.classifier == clf].set_index('feature_set').loc[fs]
            off = (-0.21, -0.07, 0.07, 0.21)[2 * k + j]
            ax.errorbar(xb + off, 100 * d[f'OA_{yr}'], yerr=100 * Z95 * d[f'OA_{yr}_se'], fmt=mk, ms=4, color=col,
                        mfc=col if j == 0 else 'white', capsize=1.5, lw=0.8, label=f'{clf} {yr}')
    ax.set_xticks(xb); ax.set_xticklabels(fs); ax.set_ylabel('Overall accuracy (%)'); ax.set_ylim(60, 90)
    ax.set_xlabel('Feature set')
    ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=4, loc='lower center', bbox_to_anchor=(0.31, -0.01),
               columnspacing=0.9, handletextpad=0.2)
    ax.set_title('(a) Classifier and feature set', loc='left', pad=3)
    ax = axs[1]; pi = pi.iloc[::-1]
    ax.barh(np.arange(len(pi)), pi.mean_drop_pp, xerr=pi.sd_drop_pp, color='#3a7dc9', height=0.6, capsize=2, error_kw=dict(lw=0.7))
    ax.set_yticks(np.arange(len(pi))); ax.set_yticklabels([f'{g} ({n})' for g, n in zip(pi.group, pi.n_features)])
    ax.set_xlabel('Decrease in overall\naccuracy (pp)'); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title('(b) Grouped permutation importance', loc='right', pad=3)
    for a_ in axs:
        for s in ('top', 'right'):
            a_.spines[s].set_visible(False)
    fig.savefig(f'{FIGS}/fig10_experiments.png'); plt.close(fig)


# ---------------------------------------------------------------- Supplementary ------------
def figS1():
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.3), gridspec_kw={'wspace': 0.04, 'hspace': 0.14})
    for i, y in enumerate(YEARS):
        axs[i, 0].imshow(rgb(y), extent=EXT, interpolation='bilinear')
        axs[i, 1].imshow(rgb(y, 'spring', ('B8', 'B4', 'B3'), 2.4), extent=EXT, interpolation='bilinear')
        cnt = read(f'{PROC}/S2_{y}_obscount.tif')[0].astype(float); cnt[cnt < 0] = np.nan
        im = axs[i, 2].imshow(cnt[0] + cnt[1], cmap='viridis', vmin=0, vmax=40, extent=EXT, interpolation='nearest')
        for j, t in enumerate((f'{y} summer true colour', f'{y} spring false colour', f'{y} valid observations')):
            axs[i, j].set_title(t, fontsize=7.5, loc='left'); noticks(axs[i, j])
    cax = fig.add_axes([0.915, 0.15, 0.01, 0.7]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6)
    cb.set_label('Spring + summer observations', fontsize=6.5)
    fig.savefig(f'{FIGS}/figS1_composites.png'); plt.close(fig)


def figS2():
    import reference_chips as R
    from PIL import ImageDraw
    pick = []
    for tt in ('Low vegetation → Built-up', 'Low vegetation → Bare/sparse', 'Bare/sparse → Built-up', 'No change'):
        pick += list(sp[(sp.ref_transition == tt) & (sp.confidence == 3)].pid.head(2 if tt != 'No change' else 2))
    feats = {y: read(f'{PROC}/S2_{y}_features.tif') for y in YEARS}
    fig, axs = plt.subplots(len(pick), 4, figsize=(5.4, 1.42 * len(pick)), gridspec_kw={'wspace': 0.04, 'hspace': 0.18})
    for i, pid in enumerate(pick):
        p = sp[sp.pid == pid].iloc[0]
        ims = [R.vhr_chip(4756, p.x_utm36n, p.y_utm36n), R.s2_chip(feats[y1][0], feats[y1][2], p.row, p.col, ['B4_summer', 'B3_summer', 'B2_summer'], 3.2),
               R.vhr_chip(26334, p.x_utm36n, p.y_utm36n), R.s2_chip(feats[y2][0], feats[y2][2], p.row, p.col, ['B4_summer', 'B3_summer', 'B2_summer'], 3.2)]
        for j, im in enumerate(ims):
            axs[i, j].imshow(im); noticks(axs[i, j])
            if i == 0:
                axs[i, j].set_title((f'VHR {y1}', f'S2 {y1}', f'VHR {y2}', f'S2 {y2}')[j], fontsize=7.5)
        axs[i, 0].set_ylabel(f'{CLASSES[p.ref_2018]} to\n{CLASSES[p.ref_2025]}' if p.ref_2018 != p.ref_2025 else f'{CLASSES[p.ref_2018]}\n(no change)', fontsize=6.3)
    fig.text(0.5, 0.06, 'VHR: Esri World Imagery Wayback; Maxar, Vantor. Chips 120 m (VHR) and 250 m (S2); the 10 m pixel is outlined.',
             ha='center', fontsize=5.8)
    fig.savefig(f'{FIGS}/figS2_reference_examples.png'); plt.close(fig)


if __name__ == '__main__':
    import sys
    def ttype(a, b):
        if a == b:
            return 'No change'
        lab = f'{CLASSES[a]} → {CLASSES[b]}'
        return lab if lab in ('Low vegetation → Built-up', 'Low vegetation → Bare/sparse', 'Bare/sparse → Low vegetation',
                              'Bare/sparse → Built-up') else 'Other change'
    sp['ref_transition'] = [ttype(a, b) for a, b in zip(sp.ref_2018, sp.ref_2025)]
    todo = sys.argv[1:] or ['fig01', 'fig02', 'fig03', 'fig04', 'fig05', 'fig06', 'fig07', 'fig08', 'fig09', 'fig10', 'figS1', 'figS2']
    for f in todo:
        with plt.rc_context(BIG if f in ('fig04', 'fig06', 'fig07', 'fig10') else {}):
            globals()[f]()
        print('done', f, flush=True)




