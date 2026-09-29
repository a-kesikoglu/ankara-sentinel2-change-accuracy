"""Design-based estimators for stratified random sampling (Cochran 1977; Stehman 2014;
Olofsson et al. 2014). The strata need not coincide with the classes of the map evaluated."""
import numpy as np, pandas as pd

Z95 = 1.96


class Design:
    def __init__(self, stratum, Nh):
        self.h = np.asarray(stratum)
        self.Nh = {int(k): float(v) for k, v in Nh.items()}
        self.N = sum(self.Nh.values())
        self.nh = {h: int((self.h == h).sum()) for h in self.Nh}

    def total(self, y):
        y = np.asarray(y, float)
        return sum(self.Nh[h] * y[self.h == h].mean() for h in self.Nh if self.nh[h] > 0)

    def proportion(self, y):
        y = np.asarray(y, float)
        p = self.total(y) / self.N
        v = sum((self.Nh[h] / self.N) ** 2 * (1 - self.nh[h] / self.Nh[h]) * y[self.h == h].var(ddof=1) / self.nh[h]
                for h in self.Nh if self.nh[h] > 1)
        return p, np.sqrt(v)

    def ratio(self, y, x):
        y = np.asarray(y, float); x = np.asarray(x, float)
        X = self.total(x)
        if X == 0:
            return np.nan, np.nan
        R = self.total(y) / X
        v = 0.0
        for h in self.Nh:
            m = self.h == h
            if self.nh[h] < 2:
                continue
            sy, sx = y[m].var(ddof=1), x[m].var(ddof=1)
            sxy = np.cov(y[m], x[m], ddof=1)[0, 1]
            v += self.Nh[h] ** 2 * (1 - self.nh[h] / self.Nh[h]) * (sy + R ** 2 * sx - 2 * R * sxy) / self.nh[h]
        return R, np.sqrt(v) / X


def accuracy(design, mapc, refc, classes):
    """Overall, user's and producer's accuracy with standard errors, plus the
    estimated error matrix in area proportions (rows = map, columns = reference)."""
    mapc, refc = np.asarray(mapc), np.asarray(refc)
    oa = design.proportion(mapc == refc)
    rows = []
    for k in classes:
        ua = design.ratio((mapc == k) & (refc == k), mapc == k)
        pa = design.ratio((mapc == k) & (refc == k), refc == k)
        f1 = 2 * ua[0] * pa[0] / (ua[0] + pa[0]) if ua[0] + pa[0] > 0 else np.nan
        rows.append({'class': k, 'n_map': int((mapc == k).sum()), 'n_ref': int((refc == k).sum()),
                     'UA': ua[0], 'UA_se': ua[1], 'PA': pa[0], 'PA_se': pa[1], 'F1': f1})
    em = np.array([[design.proportion((mapc == i) & (refc == j))[0] for j in classes] for i in classes])
    return oa, pd.DataFrame(rows), em


def area(design, refc, k):
    p, se = design.proportion(np.asarray(refc) == k)
    return p * design.N, se * design.N
