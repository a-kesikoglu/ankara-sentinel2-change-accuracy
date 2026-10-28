# -*- coding: utf-8 -*-
"""Print-size control for figures: every text element must be at least MIN_PT at the width the figure
occupies in the journal page (PRINT_CM). Fonts below the floor are enlarged before saving, and the
smallest printed font size of each figure is logged."""
import json, os
import matplotlib.figure as mfig
import matplotlib.text as mtext

PRINT_CM = 15.2          # figure width in the manuscript (full text width of the template)
MIN_PT = 8.0             # minimum printed font size
MIN_FIG = {'fig04_maps_change': 9.0, 'fig06_examples': 9.0, 'fig07_accuracy': 9.0, 'fig10_experiments': 9.0}   # dense figures, larger floor
LOG = {}
_orig_save = mfig.Figure.savefig


def _texts(fig):
    return [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip()]


def _scale(fig):
    r = fig.canvas.get_renderer()
    w = fig.get_tightbbox(r).width            # inches, as cropped by bbox_inches='tight'
    return (PRINT_CM / 2.54) / w


def _save(self, fname, *a, **k):
    name = os.path.splitext(os.path.basename(str(fname)))[0]
    lo = MIN_FIG.get(name, MIN_PT)
    for _ in range(10):                        # enlarging text can widen the tight bbox
        s = _scale(self)
        small = [t for t in _texts(self) if t.get_fontsize() * s < lo - 1e-6]
        if not small:
            break
        for t in small:
            t.set_fontsize((lo + 0.02) / s)
    s = _scale(self)
    LOG[name] = round(min(t.get_fontsize() for t in _texts(self)) * s, 2)
    out = _orig_save(self, fname, *a, **k)
    path = os.path.join(os.path.dirname(str(fname)), 'print_font_sizes.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(LOG); json.dump(old, open(path, 'w'), indent=1)
    return out


def enable():
    mfig.Figure.savefig = _save
