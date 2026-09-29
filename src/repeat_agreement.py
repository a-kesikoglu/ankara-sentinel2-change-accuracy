"""Agreement between the original and the blind repeat interpretation of a random 20% subset."""
import json
import numpy as np, pandas as pd
from common import *

key = pd.read_csv(f'{SAMP}/repeat_key.csv')
rep = pd.read_csv(f'{SAMP}/repeat_labels.csv').merge(key, on='rid')
orig = pd.read_csv(f'{SAMP}/reference_labels.csv')
m = rep.merge(orig, on='pid', suffixes=('_rep', '_orig'))
a18 = (m.ref_2018_rep == m.ref_2018_orig); a25 = (m.ref_2025_rep == m.ref_2025_orig)
ch_rep = m.ref_2018_rep != m.ref_2025_rep; ch_orig = m.ref_2018_orig != m.ref_2025_orig
res = {'n': len(m), 'agree_2018': float(a18.mean()), 'agree_2025': float(a25.mean()),
       'agree_both_dates': float((a18 & a25).mean()), 'agree_change_nochange': float((ch_rep == ch_orig).mean()),
       'n_change_orig': int(ch_orig.sum()), 'n_change_rep': int(ch_rep.sum()),
       'agree_by_confidence': {int(c): float((a18 & a25)[m.confidence == c].mean()) for c in (1, 2, 3)}}
json.dump(res, open(f'{TABS}/repeat_interpretation_agreement.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
print(m[~(a18 & a25)][['rid', 'pid', 'ref_2018_orig', 'ref_2025_orig', 'ref_2018_rep', 'ref_2025_rep', 'confidence']])
