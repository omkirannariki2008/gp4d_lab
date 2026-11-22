#!/usr/bin/env python3
"""
tools/root_to_csv.py — Python ROOT → CSV Converter  (fixed v2)
================================================================
Converts 4D ROOT histograms (THn / THnSparse / nested TH3) from the
Sartre inclusive-diffraction lookup-table format into flat CSVs ready
for gp4d training.

Fixes vs v1
-----------
* axis.name / axis.title use .member("fName") / .member("fTitle")  (uproot API)
* axis.min / axis.max guarded with .member("fXmin") / .member("fXmax") fallback
* h.values(flow=False) — exclude under/overflow bins
* THnSparse support via h.to_numpy() path
* Handles both TH3D-family and THnSparse objects
* Removes the dead nested-loop (was building `rows` but never using them)
* Correct beta/z column in the vectorised DataFrame build (was copying Q2 values)
* Better axis-name matching for Sartre files (Q2/W2/beta/zeta in fTitle)

Usage
-----
    python tools/root_to_csv.py -i root_files/ -o data/
    python tools/root_to_csv.py --config configs/default_config.toml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

# ── uproot axis helpers ──────────────────────────────────────────────────────

def _axis_member(axis, key: str, default=""):
    """Safely read a ROOT member from an uproot TAxis."""
    try:
        return axis.member(key)
    except Exception:
        pass
    try:
        return getattr(axis, key.lstrip("f").lower(), default)
    except Exception:
        return default


def _axis_name(axis) -> str:
    return str(_axis_member(axis, "fName", "")).lower()


def _axis_title(axis) -> str:
    return str(_axis_member(axis, "fTitle", "")).lower()


def _axis_min(axis) -> float:
    for key in ("fXmin", "min"):
        try:
            v = axis.member(key) if key.startswith("f") else getattr(axis, key)
            return float(v)
        except Exception:
            pass
    try:
        return float(axis.edges()[0])
    except Exception:
        return 0.0


def _axis_max(axis) -> float:
    for key in ("fXmax", "max"):
        try:
            v = axis.member(key) if key.startswith("f") else getattr(axis, key)
            return float(v)
        except Exception:
            pass
    try:
        return float(axis.edges()[-1])
    except Exception:
        return 1.0


# ── log-scale detection ──────────────────────────────────────────────────────

def _detect_log_scale(axis, expected_lin_max: float, var_hint: str = "") -> bool:
    """
    Decide whether this axis stores values already in log space.
    Logic mirrors the original C++ detect_log_scale_for_var.
    """
    # Explicitly check for beta and z/zeta, which are always linear
    vh = var_hint.lower()
    if "beta" in vh or "zeta" in vh or vh == "z":
        return False

    xmin = _axis_min(axis)
    xmax = _axis_max(axis)

    # Negative minimum ⟹ must be in log space (for Q2/W2 log scales)
    # But only if it's significantly negative, to avoid noise/boundaries.
    if xmin < -0.1:
        return True

    # Check the ROOT axis name and title for explicit "log" keyword
    combined = _axis_name(axis) + " " + _axis_title(axis) + " " + var_hint.lower()
    if "log" in combined:
        return True

    # Numeric heuristic: which interpretation puts xmax closer to the expected range?
    if expected_lin_max > 0:
        expected_log_max = np.log(max(expected_lin_max, EPS))
        diff_lin = abs(xmax - expected_lin_max)
        diff_log = abs(xmax - expected_log_max)
        return diff_log < diff_lin

    return False


# ── axis edge/center extraction ──────────────────────────────────────────────

def _get_edges(axis) -> np.ndarray:
    """Return finite bin edges (no ±inf flow)."""
    try:
        edges = axis.edges(flow=False)
    except TypeError:
        edges = axis.edges()
    edges = np.asarray(edges, dtype=float)
    # Strip ±inf overflow that some uproot versions include
    edges = edges[np.isfinite(edges)]
    return edges


def _prepare_axis(centers: np.ndarray, edges: np.ndarray, is_log: bool) -> np.ndarray:
    """
    Returns shape (n, 6): lin_center, lin_low, lin_high, log_center, log_low, log_high
    """
    out = np.empty((len(centers), 6), dtype=float)
    for i, (c, lo, hi) in enumerate(zip(centers, edges[:-1], edges[1:])):
        if is_log:
            out[i] = (np.exp(c), np.exp(lo), np.exp(hi), c, lo, hi)
        else:
            lc  = np.log(max(abs(c),  EPS))
            ll  = np.log(max(abs(lo), EPS))
            lh  = np.log(max(abs(hi), EPS))
            out[i] = (c, lo, hi, lc, ll, lh)
    return out


# ── histogram unpacking ──────────────────────────────────────────────────────

def _unpack_histogram(h):
    """
    Return (axes_list, values_4d_array) regardless of whether h is
    TH3D (with 4th dim encoded differently), THnD, THnSparseD, or THnT.

    For Sartre tables the histogram is usually a THnSparseD or THnT with 4 axes.
    """
    class_name = type(h).__name__

    # ── THnT / dynamic THn with fAxes/fArray but no .axes path ───────────
    if not hasattr(h, "axes") and h.has_member("fAxes") and h.has_member("fArray"):
        axes = list(h.member("fAxes"))
        arr = h.member("fArray")
        dims = [a.member("fNbins") + 2 for a in axes]
        data = np.asarray(arr.member("fData"), dtype=float)
        data_reshaped = data.reshape(dims)
        slices = tuple(slice(1, -1) for _ in axes)
        vals_nd = data_reshaped[slices]
        return axes, vals_nd

    # ── THn / THnSparse path ─────────────────────────────────────────────
    if "THnSparse" in class_name or "THn" in class_name:
        # uproot exposes axes as h.axes (a list of TAxis) for THn family
        axes = list(h.axes)
        # to_numpy() returns (values_dense, [edges_ax0, edges_ax1, ...])
        try:
            vals_nd, _ = h.to_numpy(flow=False)
        except TypeError:
            vals_nd, _ = h.to_numpy()
        vals_nd = np.asarray(vals_nd, dtype=float)
        return axes, vals_nd

    # ── Standard TH (TH2D / TH3D treated as 4D via stacking) ────────────
    # uproot TH2/TH3 expose .axes as a tuple of TAxis
    axes = list(h.axes)
    try:
        vals_nd = np.asarray(h.values(flow=False), dtype=float)
    except TypeError:
        vals_nd = np.asarray(h.values(), dtype=float)

    return axes, vals_nd


# ── main conversion ──────────────────────────────────────────────────────────

def process_file(
    root_path: Path,
    output_dir: Path,
    filter_zero: bool = True,
    zero_thresh: float = EPS,
    expected_Q2_max: float = 100.0,
    expected_W2_max: float = 19881.0,
    histogram_key: str = "table",
    verbose: bool = True,
):
    """Convert one .root file → <stem>.csv and <stem>_logA0.csv."""
    try:
        import uproot
    except ImportError:
        sys.exit("ERROR: uproot not installed.  pip install uproot")

    stem     = root_path.stem
    csv_all  = output_dir / f"{stem}.csv"
    csv_nz   = output_dir / f"{stem}_logA0.csv"

    t0 = time.perf_counter()
    if verbose:
        print(f"\n▶  {root_path.name}")

    try:
        f = uproot.open(str(root_path))
    except Exception as e:
        print(f"  ✗ Cannot open: {e}", file=sys.stderr); return

    # ── Find histogram ────────────────────────────────────────────────────
    key = histogram_key
    if key not in f:
        # try case-insensitive scan
        for k in f.keys():
            if k.split(";")[0].lower() == key.lower():
                key = k.split(";")[0]; break
        else:
            print(f"  ✗ '{histogram_key}' not found. Keys: {f.keys()}", file=sys.stderr)
            return

    h = f[key]

    # ── Unpack ────────────────────────────────────────────────────────────
    try:
        axes, vals = _unpack_histogram(h)
    except Exception as e:
        print(f"  ✗ Failed to unpack histogram: {e}", file=sys.stderr); return

    if len(axes) != 4:
        print(f"  ✗ Expected 4 axes, got {len(axes)}: {[_axis_name(a) for a in axes]}",
              file=sys.stderr); return

    ax_Q2, ax_W2, ax_beta, ax_z = axes

    # ── Detect log scales ─────────────────────────────────────────────────
    logQ2   = _detect_log_scale(ax_Q2,   expected_Q2_max,   "Q2")
    logW2   = _detect_log_scale(ax_W2,   expected_W2_max,   "W2")
    logBeta = _detect_log_scale(ax_beta, 0.999,             "beta")
    logZ    = _detect_log_scale(ax_z,    0.999,             "zeta z")

    if verbose:
        print(f"  Detected log-scale: Q2={logQ2}  W2={logW2}  beta={logBeta}  z={logZ}")
        names = [_axis_name(a) or f"ax{i}" for i, a in enumerate(axes)]
        titles = [_axis_title(a) for a in axes]
        print(f"  Axis names : {names}")
        print(f"  Axis titles: {titles}")

    # ── Extract bin geometry ──────────────────────────────────────────────
    eQ2   = _get_edges(ax_Q2);   cQ2   = (eQ2[:-1]  + eQ2[1:])  / 2
    eW2   = _get_edges(ax_W2);   cW2   = (eW2[:-1]  + eW2[1:])  / 2
    eBeta = _get_edges(ax_beta); cBeta = (eBeta[:-1] + eBeta[1:]) / 2
    eZ    = _get_edges(ax_z);    cZ    = (eZ[:-1]   + eZ[1:])   / 2

    nQ2, nW2, nBeta, nZ = len(cQ2), len(cW2), len(cBeta), len(cZ)
    total = nQ2 * nW2 * nBeta * nZ

    if verbose:
        print(f"  Grid: {nQ2}×{nW2}×{nBeta}×{nZ} = {total:,} bins")
        print(f"  vals shape: {vals.shape}")

    # Validate shape matches edges
    expected_shape = (nQ2, nW2, nBeta, nZ)
    if vals.shape != expected_shape:
        print(f"  ⚠ Shape mismatch: vals={vals.shape} expected={expected_shape}",
              file=sys.stderr)
        # Try to trim to the smaller of the two
        s = tuple(min(a, b) for a, b in zip(vals.shape, expected_shape))
        vals = vals[:s[0], :s[1], :s[2], :s[3]]
        cQ2   = cQ2[:s[0]];   eQ2   = eQ2[:s[0]+1]
        cW2   = cW2[:s[1]];   eW2   = eW2[:s[1]+1]
        cBeta = cBeta[:s[2]]; eBeta = eBeta[:s[2]+1]
        cZ    = cZ[:s[3]];    eZ    = eZ[:s[3]+1]
        nQ2, nW2, nBeta, nZ = s

    # ── Per-axis (linc, linl, linh, logc, logl, logh) ────────────────────
    aQ2   = _prepare_axis(cQ2,   eQ2,   logQ2)
    aW2   = _prepare_axis(cW2,   eW2,   logW2)
    aBeta = _prepare_axis(cBeta, eBeta, logBeta)
    aZ    = _prepare_axis(cZ,    eZ,    logZ)

    # ── Vectorised DataFrame build ────────────────────────────────────────
    iQ2, iW2, iBeta, iZ = np.meshgrid(
        np.arange(nQ2), np.arange(nW2), np.arange(nBeta), np.arange(nZ),
        indexing="ij",
    )
    iQ2   = iQ2.ravel()
    iW2   = iW2.ravel()
    iBeta = iBeta.ravel()
    iZ    = iZ.ravel()

    df = pd.DataFrame({
        # ── Primary training features (logged where physics says so) ──
        "Q2_log_center":   aQ2[iQ2,   3],   # log(Q2) center
        "W2_log_center":   aW2[iW2,   3],   # log(W2) center
        "beta_center":     aBeta[iBeta, 0],  # beta linear center  [0,1]
        "z_center":        aZ[iZ,     0],   # zeta linear center  [0,1]
        # ── Target ────────────────────────────────────────────────────
        "logA":            vals.ravel(),
        # ── Bin edges for reference / uncertainty estimation ──────────
        "Q2_log_low":      aQ2[iQ2,   4],
        "Q2_log_high":     aQ2[iQ2,   5],
        "W2_log_low":      aW2[iW2,   4],
        "W2_log_high":     aW2[iW2,   5],
        "beta_low":        aBeta[iBeta, 1],
        "beta_high":       aBeta[iBeta, 2],
        "z_low":           aZ[iZ,     1],
        "z_high":          aZ[iZ,     2],
        # ── Linear values for plotting / post-processing ──────────────
        "Q2_lin_center":   aQ2[iQ2,   0],
        "W2_lin_center":   aW2[iW2,   0],
        "beta_log_center": aBeta[iBeta, 3],
        "z_log_center":    aZ[iZ,     3],
        # ── Bin index (useful for Q2-partition training) ───────────────
        "Q2_bin_idx":      iQ2.astype(np.int32),
    })

    # ── Write ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_all, index=False, float_format="%.10g")
    if verbose:
        print(f"  ✔ Full grid  → {csv_all}  ({len(df):,} rows)")

    if filter_zero:
        mask = np.abs(df["logA"]) > zero_thresh
        df_nz = df[mask]
        df_nz.to_csv(csv_nz, index=False, float_format="%.10g")
        if verbose:
            print(f"  ✔ Non-zero   → {csv_nz}  ({len(df_nz):,} rows, "
                  f"{len(df)-len(df_nz):,} zero bins dropped)")

    if verbose:
        print(f"  ⏱  {time.perf_counter()-t0:.2f}s")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Convert Sartre 4D ROOT lookup tables to CSV."
    )
    p.add_argument("-i", "--input",   default=None,    help="Directory of .root files")
    p.add_argument("-o", "--output",  default=None,    help="Output directory for CSVs")
    p.add_argument("-k", "--key",     default="table", help="Histogram key inside the ROOT file")
    p.add_argument("--no-filter-zero", action="store_true",
                   help="Keep rows where logA≈0")
    p.add_argument("--config", default=None,
                   help="TOML config (reads [root_converter] section)")
    args = p.parse_args()

    input_dir   = args.input
    output_dir  = args.output
    filter_zero = not args.no_filter_zero
    zero_thresh = EPS
    expected_Q2_max = 100.0
    expected_W2_max = 19881.0
    hkey = args.key

    if args.config:
        try:
            try:   import tomllib
            except ImportError: import tomli as tomllib
            with open(args.config, "rb") as f:
                cfg = tomllib.load(f)
            rc = cfg.get("root_converter", {})
            input_dir       = input_dir  or rc.get("input_dir")
            output_dir      = output_dir or rc.get("output_dir")
            filter_zero     = rc.get("filter_zero_target", filter_zero)
            zero_thresh     = rc.get("zero_threshold", zero_thresh)
            expected_Q2_max = rc.get("expected_Q2_max", expected_Q2_max)
            expected_W2_max = rc.get("expected_W2_max", expected_W2_max)
            hkey            = rc.get("histogram_key", hkey)
        except Exception as e:
            print(f"Warning: config load failed: {e}", file=sys.stderr)

    if not input_dir or not output_dir:
        p.print_help(); sys.exit(1)

    input_path = Path(input_dir)
    if input_path.is_file():
        root_files = [input_path]
    else:
        root_files = sorted(input_path.glob("*.root"))

    if not root_files:
        sys.exit(f"No .root files in or matching: {input_dir}")

    print(f"Found {len(root_files)} .root file(s)")
    for rf in root_files:
        process_file(
            root_path       = rf,
            output_dir      = Path(output_dir),
            filter_zero     = filter_zero,
            zero_thresh     = zero_thresh,
            expected_Q2_max = expected_Q2_max,
            expected_W2_max = expected_W2_max,
            histogram_key   = hkey,
        )
    print(f"\n✔ Done → {output_dir}")


if __name__ == "__main__":
    main()
