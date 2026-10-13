"""Choose (sample, feature pair) slices that are likely to make good figures.

A snapshot slice is a good figure when
  (1) both classes occupy a real part of the plane,
  (2) every method's explanation is visible (not empty, not a sliver), and
  (3) the explanations reach towards the decision boundary.

Two stages, both per network:

**Stage 1 -- analysis (no rendering).**
  * dataset: binary / one-hot features are dropped as axes
    (:func:`common.axis_candidates`); ordinal features are mildly penalised.
  * network: the logit ``z`` and its input gradient ``g`` at every sample.  A
    linearisation predicts whether the boundary crosses the slice, and how
    evenly it splits it.
  * explanations: the 1D extent of each explanation through the sample point
    along every axis.  A feature that an explanation fixes has extent 0 and
    gives a degenerate slice; a long extent towards the (linearised) boundary
    is what makes an explanation look "close to the boundary".
  These combine into a ``prior`` score per (sample, pair).

**Stage 2 -- exact slice metrics.**  For the full box and a few zoomed
windows around the explanations, on a 100x100 grid over the plotted window: the class map from the network, the exact snapshot polygon of each
method, and from them balance, size, boundary closeness, soundness and how
different the methods look.  This is cheap here (seconds per network), so it
runs on every candidate and the prior is reported as a validated shortcut
(Spearman correlation, recall of the exact top-K) for larger settings.

Writes, per ``plots/tacas/<dataset>/<net>/``:
  ``features.csv``  per-feature network sensitivity and explanation freedom
  ``scores.csv``    every (sample, pair) with all metrics
  ``selection.csv`` the diverse top-K that ``render.py`` draws

    PYTHONPATH=. python3 -m explviz.examples.tacas.select_pairs [--top 25]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

from explviz.geometry.halfplanes import close_ring, polygon_area
from explviz.geometry.region2d import regions_for_formula

from .common import (DATASETS, METHODS, NETWORKS, OUT_DIR, axis_candidates,
                     load_classifier, load_points_for, load_polytopes)

RES = 100                 # grid resolution of the exact metrics
NEAR = 0.04               # "touching the boundary" tolerance, in scaled units
ORDINAL_LEVELS = 10       # fewer distinct values than this -> ordinal feature
ORDINAL_PENALTY = 0.85
MIN_BALANCE = 0.10        # minority class must cover >= 5% of the plane
#: Plot windows tried per slice: "full" is the [0,1]^2 box, "zoomK" the
#: explanations' bounding box enlarged K times, "bnd" frames the explanations,
#: the sample and the nearest decision-boundary point (padded by BND_PAD).
#: 50x4 explanations are ~5x narrower than 50x1 ones: specks in the full box.
VIEWS = ("full", "zoom2.5", "zoom4", "zoom8", "bnd")
MIN_SPAN = 0.05
BND_PAD = 1.6
ZOOM_PENALTY = 0.97       # a zoomed view must be strictly better to win


# --------------------------------------------------------------------------
# Stage 1: analysis
# --------------------------------------------------------------------------

def logit_and_grad(clf, X):
    x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    z = clf.module(x).squeeze(-1)
    z.sum().backward()
    return z.detach().numpy().astype(float), x.grad.numpy().astype(float)


def extent_1d(polys, x, j, lo=0.0, hi=1.0):
    """Interval [t-, t+] such that x + t e_j stays in the explanation and box."""
    tmin, tmax = lo - x[j], hi - x[j]
    best = None
    for p in polys:
        s = p.b_ub - p.A_ub @ x
        if np.any(s < -1e-9):
            continue                      # this disjunct does not hold x
        aj = p.A_ub[:, j]
        pos, neg = aj > 1e-12, aj < -1e-12
        up = min([tmax] + list(s[pos] / aj[pos]))
        dn = max([tmin] + list(s[neg] / aj[neg]))
        if p.A_eq.shape[0] and np.any(np.abs(p.A_eq[:, j]) > 1e-12):
            up = dn = 0.0                 # an equality pins x_j
        if best is None or up - dn > best[1] - best[0]:
            best = (dn, max(up, dn))
    return best or (0.0, 0.0)


def linear_balance(z, g, x, a, b, n=25):
    """Minority-class share of the plane under the linearised network."""
    u = np.linspace(0, 1, n)
    uu, vv = np.meshgrid(u, u)
    lin = z + g[a] * (uu - x[a]) + g[b] * (vv - x[b])
    p = np.mean(lin > 0)
    return min(p, 1 - p) / 0.5


def analysis(X, z, g, polys_by_method, axes):
    """Per-sample, per-feature quantities and the per-pair prior."""
    n_samples = len(X)
    methods = list(polys_by_method)
    width = {m: np.full((n_samples, X.shape[1]), np.nan) for m in methods}
    reach = {m: np.full((n_samples, X.shape[1]), np.nan) for m in methods}

    # direction along each axis that moves the logit towards 0, and the room
    # to do so inside the box; delta = linearised distance to the boundary
    toward = -np.sign(z[:, None] * g)
    room = np.where(toward > 0, 1 - X, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.minimum(np.abs(z)[:, None] / np.abs(g), room)
        cross = np.abs(g) * room / np.abs(z)[:, None]   # >= 1: single-axis crossing

    for m, polys in polys_by_method.items():
        for i in range(n_samples):
            if polys[i] is None:
                continue
            for j in axes:
                dn, up = extent_1d(polys[i], X[i], j)
                width[m][i, j] = up - dn
                ext = up if toward[i, j] > 0 else -dn
                reach[m][i, j] = 1.0 if delta[i, j] <= 1e-9 else min(1.0, ext / delta[i, j])

    return width, reach, cross


def prior_score(i, a, b, X, z, g, width, reach, penalty, methods):
    bal = linear_balance(z[i], g[i], X[i], a, b)
    per = []
    for m in methods:
        wa, wb = width[m][i, a], width[m][i, b]
        size = np.sqrt(min(1, wa / 0.4) * min(1, wb / 0.4))
        per.append(size * 0.5 * (reach[m][i, a] + reach[m][i, b]))
    return np.sqrt(bal) * float(np.mean(per)) * penalty[a] * penalty[b]


# --------------------------------------------------------------------------
# Stage 2: exact slice metrics
# --------------------------------------------------------------------------

_U = np.linspace(0, 1, RES)
_UU, _VV = np.meshgrid(_U, _U)
_GRID_PTS = np.column_stack([_UU.ravel(), _VV.ravel()])


def class_maps(clf, x, pairs, windows):
    """Predicted class (bool, class 1) on each pair's window grid, one forward pass."""
    X = np.tile(x, (len(pairs) * RES * RES, 1))
    for k, ((a, b), (alo, ahi, blo, bhi)) in enumerate(zip(pairs, windows)):
        sl = slice(k * RES * RES, (k + 1) * RES * RES)
        X[sl, a] = alo + (ahi - alo) * _UU.ravel()
        X[sl, b] = blo + (bhi - blo) * _VV.ravel()
    return (clf.logits(X) > 0).reshape(len(pairs), RES, RES)


FULL = (0.0, 1.0, 0.0, 1.0)


def boundary_points(cmap):
    """Decision-boundary cells of a full-box class map, in [0,1]^2 coordinates."""
    edge = np.zeros_like(cmap)
    edge[:, 1:] |= cmap[:, 1:] != cmap[:, :-1]
    edge[1:, :] |= cmap[1:, :] != cmap[:-1, :]
    iy, ix = np.nonzero(edge)
    return np.column_stack([ix, iy]) / (RES - 1)


def window_for(regions, x, a, b, view, full_cmap=None):
    """(a_lo, a_hi, b_lo, b_hi) of the plot window, inside [0,1]^2; None if n/a."""
    if view == "full":
        return FULL
    pts = np.vstack([r for rs in regions.values() for r in rs] + [[[x[a], x[b]]]])
    if view == "bnd":
        bpts = boundary_points(full_cmap) if full_cmap is not None else []
        if not len(bpts):
            return None
        near = bpts[np.argmin(np.linalg.norm(bpts - [x[a], x[b]], axis=1))]
        pts = np.vstack([pts, near])
        zoom = BND_PAD
    else:
        zoom = float(view[len("zoom"):])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.clip((hi - lo) * zoom, MIN_SPAN, 1.0)
    wlo = np.clip((lo + hi) / 2 - span / 2, 0.0, 1.0 - span)
    whi = wlo + span
    return (wlo[0], whi[0], wlo[1], whi[1])


def to_unit(regions, window):
    """Map polygon vertices from feature space into [0,1]^2 window coordinates."""
    alo, ahi, blo, bhi = window
    lo, span = np.array([alo, blo]), np.array([ahi - alo, bhi - blo])
    return {m: [(r - lo) / span for r in rs] for m, rs in regions.items()}


def boundary_distance(cmap):
    """Distance (scaled units) from every grid cell to the decision boundary."""
    edge = np.zeros_like(cmap)
    edge[:, 1:] |= cmap[:, 1:] != cmap[:, :-1]
    edge[:, :-1] |= cmap[:, 1:] != cmap[:, :-1]
    edge[1:, :] |= cmap[1:, :] != cmap[:-1, :]
    edge[:-1, :] |= cmap[1:, :] != cmap[:-1, :]
    if not edge.any():
        return None
    return distance_transform_edt(~edge) / (RES - 1)


def ring_samples(ring, per_edge=25):
    pts = [np.linspace(ring[k], ring[k + 1], per_edge, endpoint=False)
           for k in range(len(ring) - 1)]
    return np.vstack(pts)


def slice_metrics(cmap, own, regions_by_method):
    """Exact metrics of one slice; ``None`` if it can't make a usable figure."""
    p1 = cmap.mean()
    balance = min(p1, 1 - p1) / 0.5
    if balance < MIN_BALANCE:
        return None
    dist = boundary_distance(cmap)
    if dist is None:
        return None
    own_mask = cmap == bool(own)
    own_area = own_mask.mean()

    out, masks, quality = {"balance": balance, "own_area": own_area}, [], []
    for m, regions in regions_by_method.items():
        if not regions:
            return None                    # an explanation vanishes in this slice
        area, mask, gap, tight = 0.0, np.zeros(RES * RES, bool), np.inf, []
        for r in regions:
            area += polygon_area(r)
            mask |= MplPath(r).contains_points(_GRID_PTS, radius=1e-9)
            pts = ring_samples(close_ring(r))
            ij = np.clip(np.rint(pts * (RES - 1)).astype(int), 0, RES - 1)
            d = dist[ij[:, 1], ij[:, 0]]
            on_box = np.any((pts < 1e-6) | (pts > 1 - 1e-6), axis=1)
            gap = min(gap, d.min())
            tight.append((d < NEAR) | on_box)
        mask = mask.reshape(RES, RES)
        tight = np.concatenate(tight).mean()
        unsound = float(np.mean(cmap[mask] != bool(own))) if mask.any() else 0.0
        closeness = 0.5 * np.exp(-gap / 0.03) + 0.5 * tight
        size = np.sqrt(min(1.0, area / 0.2))
        quality.append(size * closeness * (1.0 - min(1.0, 5 * unsound)))
        masks.append(mask)
        out.update({f"{m}:area": area, f"{m}:coverage": area / max(own_area, 1e-9),
                    f"{m}:gap": gap, f"{m}:tight": tight, f"{m}:unsound": unsound,
                    f"{m}:quality": quality[-1]})

    ious = []
    for u, v in itertools.combinations(masks, 2):
        union = (u | v).sum()
        ious.append((u & v).sum() / union if union else 1.0)
    out["distinct"] = 1.0 - float(np.mean(ious)) if ious else 0.0
    out["quality"] = float(np.mean(quality))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def reselect(ds, net, top):
    """Redo only the selection from a saved ``scores.csv`` (seconds)."""
    out = variant_dir(ds) / net
    scores = pd.read_csv(out / "scores.csv")
    counts = scores.drop_duplicates("sample").methods.str.split("+").explode().value_counts()
    rare = [m for m, c in counts.items() if c < 0.1 * len(load_points_for(ds)[0])]
    sel = select_diverse(scores, top, rare)
    sel.to_csv(out / "selection.csv", index=False)
    return sel


def run_network(ds, net, top):
    X, y = load_points_for(ds)
    clf = load_classifier(ds, net)
    pred = clf.predict(X)
    z, g = logit_and_grad(clf, X)
    polys = {m: load_polytopes(ds, net, m) for m in METHODS}
    polys = {m: p for m, p in polys.items() if any(q is not None for q in p)}

    axes = axis_candidates(X)
    levels = {j: len(np.unique(X[:, j])) for j in axes}
    penalty = {j: 1.0 if levels[j] >= ORDINAL_LEVELS else ORDINAL_PENALTY for j in axes}
    pairs = list(itertools.combinations(axes, 2))
    width, reach, cross = analysis(X, z, g, polys, axes)

    # a method present for most samples must be present in every figure, so a
    # slice never silently loses e.g. Fix-all; rare ones (Direct on 50x4) are
    # added where they exist
    # (a sample may stand in a rare method for a missing one: HA 50x4 sample 31
    # has the only Direct explanation, but its Fix-all formula is unparsable)
    coverage = {m: np.mean([p is not None for p in polys[m]]) for m in polys}
    required = [m for m in polys if coverage[m] >= 0.5]
    rare = [m for m in polys if coverage[m] < 0.1]

    rows = []
    for i in range(len(X)):
        methods = [m for m in polys if polys[m][i] is not None]
        if any(polys[m][i] is None for m in required) and not set(rare) & set(methods):
            continue
        full = {(a, b): {m: regions_for_formula(polys[m][i], (a, b), ds.spec.bounds,
                                                "snapshot", point=X[i]) for m in methods}
                for a, b in pairs}
        live = [ab for ab in pairs if all(full[ab].values())]
        best, full_cmaps = {}, {}
        for view in VIEWS:
            todo = [(ab, window_for(full[ab], X[i], *ab, view, full_cmaps.get(ab)))
                    for ab in live]
            todo = [(ab, w) for ab, w in todo if w is not None]
            if not todo:
                continue
            cmaps = class_maps(clf, X[i], [ab for ab, _ in todo], [w for _, w in todo])
            for ((a, b), win), cmap in zip(todo, cmaps):
                if view == "full":
                    full_cmaps[(a, b)] = cmap
                    regions = full[(a, b)]
                else:
                    bounds = list(ds.spec.bounds)
                    bounds[a], bounds[b] = win[:2], win[2:]
                    regions = {m: regions_for_formula(polys[m][i], (a, b), bounds,
                                                      "snapshot", point=X[i]) for m in methods}
                met = slice_metrics(cmap, pred[i], to_unit(regions, win))
                if met is None:
                    continue
                met["score"] = (np.sqrt(met["balance"]) * met["quality"]
                                * (0.6 + 0.4 * met["distinct"]) * penalty[a] * penalty[b]
                                * (1.0 if view == "full" else ZOOM_PENALTY))
                if met["score"] > best.get((a, b), {}).get("score", 0.0):
                    met.update(view=view, win_a_lo=win[0],
                               win_a_hi=win[1], win_b_lo=win[2], win_b_hi=win[3])
                    best[(a, b)] = met
        for a, b in pairs:
            row = {"sample": i, "a": a, "b": b, "methods": "+".join(methods),
                   "n_methods": len(methods), "pred": int(pred[i]), "label": int(y[i]),
                   "logit": z[i],
                   "prior": prior_score(i, a, b, X, z, g, width, reach, penalty, methods)}
            row.update(best.get((a, b), {"score": 0.0}))
            rows.append(row)
    scores = pd.DataFrame(rows)

    sel = select_diverse(scores, top, rare)
    out = variant_dir(ds) / net
    out.mkdir(parents=True, exist_ok=True)
    scores.sort_values("score", ascending=False).to_csv(out / "scores.csv", index=False)
    sel.to_csv(out / "selection.csv", index=False)
    feats = feature_table(ds, X, g, width, cross, axes, levels)
    feats.to_csv(out / "features.csv", index=False)
    return scores, sel, feats, summarize(ds, net, scores, sel, pred, polys, axes, top, rare)


def select_diverse(scores, top, rare=(), per_sample=2, per_pair=4,
                   strong_frac=0.25, per_sample_strong=6):
    """Greedy top-K by score, spread over samples and pairs.

    Samples with an explanation from a ``rare`` method get their best slice in
    first -- e.g. Direct timed out on all but one HA 50x4 sample, and that one
    is the only figure showing Direct on that network.

    Weak picks (below ``strong_frac`` of the best score) are then displaced by
    further strong slices of already-chosen samples, up to
    ``per_sample_strong`` per sample: on 50x4 only one or two samples give good
    slices at all, and more views of those beat weak ones.  Where enough
    samples are good (50x1) nothing is displaced.
    """
    ok = scores[scores.score > 0].sort_values("score", ascending=False)
    has_rare = ok.methods.apply(lambda s: any(m in s.split("+") for m in rare))
    seeds = ok[has_rare].groupby("sample").head(1)
    chosen, by_sample, by_pair = [], {}, {}

    def take(r, cap):
        key = (r.a, r.b)
        if (r.Index in chosen or by_sample.get(r.sample, 0) >= cap
                or by_pair.get(key, 0) >= per_pair):
            return False
        chosen.append(r.Index)
        by_sample[r.sample] = by_sample.get(r.sample, 0) + 1
        by_pair[key] = by_pair.get(key, 0) + 1
        return True

    for r in seeds.itertuples():
        take(r, per_sample)
    for r in ok.itertuples():
        if len(chosen) >= top:
            break
        take(r, per_sample)

    strong = strong_frac * (ok.score.iloc[0] if len(ok) else 0.0)
    protected = set(seeds.index)
    weak = [c for c in chosen if scores.score[c] < strong and c not in protected]
    weak.sort(key=lambda c: scores.score[c])
    for r in ok[ok.score >= strong].itertuples():
        if not weak:
            break
        if take(r, per_sample_strong):
            drop = weak.pop(0)
            chosen.remove(drop)
            by_sample[scores["sample"][drop]] -= 1
            by_pair[(scores.a[drop], scores.b[drop])] -= 1

    sel = scores.loc[chosen].sort_values("score", ascending=False).reset_index(drop=True)
    sel.insert(0, "rank", np.arange(1, len(sel) + 1))
    return sel


def feature_table(ds, X, g, width, cross, axes, levels):
    rows = []
    for j in axes:
        row = {"feature": f"x{j + 1}", "name": ds.spec.axis_labels[j], "levels": levels[j],
               "mean_abs_grad": float(np.mean(np.abs(g[:, j]))),
               "single_axis_crossing": float(np.mean(cross[:, j] >= 1))}
        for m, w in width.items():
            col = w[:, j][~np.isnan(w[:, j])]
            if len(col):
                row[f"{m}:mean_width"] = float(col.mean())
                row[f"{m}:fixed_frac"] = float(np.mean(col < 1e-6))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_abs_grad", ascending=False)


def summarize(ds, net, scores, sel, pred, polys, axes, top, rare):
    valid = scores[scores.score > 0]
    k = min(top, len(valid))
    exact_top = set(valid.nlargest(k, "score").index)
    budget = max(k, int(0.1 * len(scores)))
    prior_top = set(scores.nlargest(budget, "prior").index)
    rho = spearmanr(scores.prior, scores.score).statistic if len(scores) > 2 else np.nan
    pair_counts = sel.groupby(["a", "b"]).size().sort_values(ascending=False)
    return {
        "dataset": ds.name, "net": net,
        "methods": {m: int(sum(p is not None for p in polys[m])) for m in polys},
        "scored_samples": int(scores["sample"].nunique()),
        "axes": [f"x{j + 1}" for j in axes],
        "candidates": int(len(scores)), "usable": int(len(valid)),
        "pred1_frac": float(np.mean(pred)),
        "spearman_prior_vs_exact": float(rho),
        "prior_budget": int(budget),
        "recall_exact_topk_in_prior_budget": len(exact_top & prior_top) / max(k, 1),
        "selected_samples": int(sel["sample"].nunique()),
        "rare_methods": rare,
        "views_in_selection": {str(k): int(v) for k, v in sel.view.value_counts().items()},
        "median_selected_score": float(sel.score.median()),
        "selected_pairs": {f"x{a + 1}-x{b + 1}": int(c) for (a, b), c in pair_counts.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--nets", nargs="+", default=NETWORKS)
    ap.add_argument("--reselect", action="store_true",
                    help="only redo the selection from the saved scores.csv")
    args = ap.parse_args()

    if args.reselect:
        for name in args.datasets:
            for net in args.nets:
                sel = reselect(DATASETS[name], net, args.top)
                print(name, net)
                print(sel[["rank", "sample", "a", "b", "methods", "view", "score"]]
                      .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        return

    summaries = []
    for name in args.datasets:
        for net in args.nets:
            _, sel, feats, summ = run_network(DATASETS[name], net, args.top)
            summaries.append(summ)
            print(json.dumps(summ, indent=1))
            print(feats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
            print(sel[["rank", "sample", "a", "b", "methods", "view", "score", "prior",
                       "balance", "quality", "distinct"]].to_string(index=False,
                                                         float_format=lambda v: f"{v:.3f}"))
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, indent=1))


if __name__ == "__main__":
    main()
