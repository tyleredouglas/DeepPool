#!/usr/bin/env python3

"""
Model validation: outputs error predictions on simulated populations and writes a CSV
with chrom, sim, pos, window_bp, Predicted_Error, and MCMC-estimated haplotype frequencies.
Also prints to stdout each time a window is recorded.
"""
import argparse
import sys
import re
import csv
import threading
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

write_lock = threading.Lock()

# ─────────────────────── helper functions ─────────────────────── #

def read_list_file(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def read_regions_file(path: str) -> list[tuple[str,int,int]]:
    regions = []
    pat = re.compile(r'^([^:]+):(\d+):(\d+)$')
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            m = pat.match(ln)
            if m:
                chrom, start, end = m.groups()
            else:
                parts = ln.split()
                if len(parts) != 3:
                    raise ValueError(f"Bad region spec: {ln}")
                chrom, start, end = parts
            regions.append((chrom, int(start), int(end)))
    return regions

# ─────────────────────── MCMC inference ─────────────────────── #

def mcmc_model(X, b):
    p     = numpyro.sample("p", dist.Dirichlet(jnp.ones(X.shape[1])))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))
    mu    = jnp.dot(X, p)
    numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=b)

def mcmc_haplotype_freq(
    X: np.ndarray, b: np.ndarray,
    num_warmup: int=80, num_samples: int=20,
    num_chains: int=1, rng_seed: int=0
) -> tuple[np.ndarray, dict[str, float]]:
    kernel = NUTS(mcmc_model)
    mcmc   = MCMC(kernel,
                  num_warmup=num_warmup,
                  num_samples=num_samples,
                  num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(rng_seed), X, b, extra_fields=("diverging","num_steps"))
    samples = mcmc.get_samples(group_by_chain=True)
    extra   = mcmc.get_extra_fields(group_by_chain=True)
    p_samps = jnp.reshape(samples["p"], (-1, X.shape[1]))
    p_np    = np.array(p_samps)
    divr    = float(jnp.mean(extra["diverging"]))
    steps   = extra["num_steps"]
    depth   = int((jnp.log2(steps + 1e-8).astype(int) + 1).max())
    return p_np, {"divergence_rate": divr, "max_tree_depth": depth}

# ──────────────────── feature extraction ───────────────────── #

def compute_effective_rank(X: np.ndarray) -> float:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_norm   = s / (s.sum() + 1e-8)
    ent      = -np.sum(s_norm * np.log(s_norm + 1e-8))
    return float(np.exp(ent))

def lsei_haplotype_estimator(
    X: np.ndarray, b: np.ndarray, lb: float = 0.0
) -> np.ndarray:
    k      = X.shape[1]
    cons   = ({'type': 'eq', 'fun': lambda p: p.sum() - 1.0},)
    bounds = [(lb, 1.0)] * k
    p0     = np.ones(k) / k
    def obj(p): return ((X.dot(p) - b)**2).sum()
    res = minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        bounds = [(0.0, 1.0)] * k
        res = minimize(obj, p0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x if res.success else np.full(k, np.nan)

def compute_avg_snpfreqs(df: pd.DataFrame, sim: str) -> np.ndarray:
    n_haps = sum(1 for c in df.columns if c.startswith("B"))
    out    = np.full(n_haps, np.nan)
    vals   = df[sim].astype(float)
    for i in range(n_haps):
        mask = (df[f"B{i+1}"] == 1)
        if mask.any():
            out[i] = vals[mask].mean()
    return out

def extract_window_features(
    window_df: pd.DataFrame,
    sim: str,
    hap_cols: list[str],
    num_warmup: int,
    num_samples: int
) -> dict[str, float] | None:
    n = len(window_df)
    if n == 0:
        return None

    start_c    = float(window_df.pos.min())
    end_c      = float(window_df.pos.max())
    window_bp  = end_c - start_c + 1
    b_vals     = window_df[sim].astype(float).values
    mean_f, std_f = b_vals.mean(), b_vals.std()
    snr        = mean_f / std_f if std_f > 0 else np.nan

    X          = window_df[hap_cols].astype(float).values
    cond_num   = np.linalg.cond(X) if X.shape[0] >= X.shape[1] else np.inf
    row_sums   = X.sum(axis=1)
    avg_row    = row_sums.mean()
    row_std    = row_sums.std()
    min_col_dist = (
        min(np.sum(X[:,i] != X[:,j])
            for i in range(X.shape[1]) for j in range(i+1, X.shape[1]))
        if X.shape[1] > 1 else 0
    )
    e_rank     = compute_effective_rank(X)

    p_samps, diag = mcmc_haplotype_freq(
        X, b_vals,
        num_warmup=num_warmup,
        num_samples=num_samples
    )

    avg_unc = float(np.std(p_samps, axis=0).mean())
    avg_skw = float(skew(p_samps, axis=0).mean())
    avg_krt = float(kurtosis(p_samps, axis=0).mean())
    sig     = float(std_f) + 1e-8
    H       = (X.T @ X) / (sig**2 + 1e-8)
    eigs    = np.linalg.eigvals(H)
    imp_grad= float(np.log(np.median(1.0 / (np.abs(eigs) + 1e-8)) + 1e-8))

    lsei = lsei_haplotype_estimator(X, b_vals)
    avgf = compute_avg_snpfreqs(window_df, sim)

    feats = {
        "start": start_c,
        "end": end_c,
        "window_bp": window_bp,
        "n_snps": float(n),
        "avg_uncertainty": avg_unc,
        "avg_skew": avg_skw,
        "avg_kurtosis": avg_krt,
        "snr": snr,
        "condition_number": cond_num,
        "min_col_distance": float(min_col_dist),
        "avg_row_sum": avg_row,
        "row_sum_std": row_std,
        "improved_gradient_sensitivity": imp_grad,
        "effective_rank": e_rank,
        "divergence_rate": diag["divergence_rate"],
        "avg_tree_depth": diag["max_tree_depth"],
    }
    for i, h in enumerate(hap_cols, start=1):
        feats[f"lsei_{h}"]        = float(lsei[i-1])
        feats[f"avg_SNPfreq_{h}"] = float(avgf[i-1])

    return feats

# ───────────────────────── model classes ─────────────────────────── #

class SNPTransformerEncoder(nn.Module):
    def __init__(self, n_haps, embed_dim=64, heads=4, layers=3):
        super().__init__()
        self.proj = nn.Linear(n_haps, embed_dim)
        blk      = nn.TransformerEncoderLayer(embed_dim, heads, batch_first=True)
        self.enc = nn.TransformerEncoder(blk, layers)
        self.pool = nn.Linear(embed_dim, 1)

    def forward(self, x):
        z = self.proj(x)
        z = self.enc(z)
        w = F.softmax(self.pool(z), dim=1)
        return (w * z).sum(dim=1)

class TabularRegressor(nn.Module):
    def __init__(self, inp, hidden=(256,128), drop=0.3):
        super().__init__()
        layers = []
        dims   = [inp] + list(hidden)
        for i in range(len(hidden)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(drop),
            ]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

class Predictor(nn.Module):
    def __init__(self, n_haps, n_tab, embed_dim=64):
        super().__init__()
        self.enc = SNPTransformerEncoder(n_haps, embed_dim)
        self.reg = TabularRegressor(embed_dim + n_tab)

    def forward(self, Xraw, Xtab):
        z = self.enc(Xraw)
        return self.reg(torch.cat([z, Xtab], dim=1))

# ───────────────────── adaptive windowing ───────────────────── #

def adaptive_windowing(
    df, sim, _region,
    model, scaler, err_thr,
    hap_cols, coarse_sizes, step_refine,
    min_snps, max_snps, step_start,
    num_warmup, num_samples,
    mid_model=None, mid_switch=None,
    low_model=None, low_switch=None,
    strict_cutoff=None, strict_step=None
):
    global output_fh, csv_writer
    start, end = int(df.pos.min()), int(df.pos.max())

    while start < end:
        best, found = None, False
        curr_thr    = strict_cutoff if strict_cutoff is not None else err_thr

        # coarse scan
        for cs in coarse_sizes:
            we  = start + cs
            win = df[(df.pos >= start) & (df.pos < we)]
            if len(win) < min_snps:
                continue

            feats = extract_window_features(
                win, sim, hap_cols,
                num_warmup, num_samples
            )
            if feats is None:
                continue
            meta = pd.DataFrame([feats])[scaler.feature_names_in_].replace([np.inf, -np.inf], np.nan)
            if meta.isna().any(axis=1).item():
                continue

            X_tab = torch.tensor(scaler.transform(meta), dtype=torch.float32)
            M     = win[hap_cols].astype(float).values
            if len(M) > max_snps:
                M = M[-max_snps:]
            else:
                M = np.vstack([np.zeros((max_snps - len(M), M.shape[1])), M])
            Xr = torch.tensor(M.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                pe = float(model(Xr, X_tab).item())
            if mid_model is not None and mid_switch is not None and pe <= mid_switch:
                with torch.no_grad():
                    pe = float(mid_model(Xr, X_tab).item())
                if low_model is not None and low_switch is not None and pe <= low_switch:
                    with torch.no_grad():
                        pe = float(low_model(Xr, X_tab).item())

            if best is None or pe < best["pe"]:
                best = {"pe": pe, "we": we, "cs": cs, "win": win, "feats": feats}
            if pe <= curr_thr:
                found = True
                break

        if not found and strict_cutoff is not None:
            start += strict_step
            continue
        if best is None:
            start += step_start
            continue

        # refinement
        cs0  = best["cs"]
        prev = [s for s in coarse_sizes if s < cs0]
        low  = prev[-1] if prev else coarse_sizes[0]
        for W in range(low, cs0 + 1, step_refine):
            we  = start + W
            win = df[(df.pos >= start) & (df.pos < we)]
            if len(win) < min_snps:
                continue

            feats = extract_window_features(
                win, sim, hap_cols,
                num_warmup, num_samples
            )
            if feats is None:
                continue
            meta = pd.DataFrame([feats])[scaler.feature_names_in_].replace([np.inf, -np.inf], np.nan)
            if meta.isna().any(axis=1).item():
                continue

            X_tab = torch.tensor(scaler.transform(meta), dtype=torch.float32)
            M     = win[hap_cols].astype(float).values
            if len(M) > max_snps:
                M = M[-max_snps:]
            else:
                M = np.vstack([np.zeros((max_snps - len(M), M.shape[1])), M])
            Xr = torch.tensor(M.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                pe = float(model(Xr, X_tab).item())
            if mid_model is not None and mid_switch is not None and pe <= mid_switch:
                with torch.no_grad():
                    pe = float(mid_model(Xr, X_tab).item())
                if low_model is not None and low_switch is not None and pe <= low_switch:
                    with torch.no_grad():
                        pe = float(low_model(Xr, X_tab).item())

            if pe < best["pe"] or pe <= err_thr:
                best = {"pe": pe, "we": we, "cs": W, "win": win, "feats": feats}
                break

        # write record
        rec = {
            "chrom": df.chrom.iloc[0],
            "sim":   sim,
            "pos":   int(best["feats"]["start"]),
            "window_bp": best["cs"],
            "Predicted_Error": best["pe"],
        }

        # MCMC haplotype-frequency estimate on chosen window
        win      = best["win"]
        X        = win[hap_cols].astype(float).values
        b        = win[sim].astype(float).values
        p_samps, _ = mcmc_haplotype_freq(
            X, b,
            num_warmup=num_warmup,
            num_samples=num_samples
        )
        p_mean   = p_samps.mean(axis=0)
        for i, h in enumerate(hap_cols):
            rec[h] = float(p_mean[i])

        # print to stdout
        print(f"Found window: pos={rec['pos']}, window_bp={rec['window_bp']}, Predicted_Error={rec['Predicted_Error']:.4f}")

        with write_lock:
            csv_writer.writerow(rec)
            output_fh.flush()

        nxt = df[df.pos >= start + step_start]
        if nxt.empty:
            break
        start = int(nxt.pos.iloc[0])

def main():
    ap = argparse.ArgumentParser(
        description=(
            "evaluate model from haplomatic-train; outputs chrom, sim, pos, window_bp, "
            "Predicted_Error, and MCMC-estimated haplotype frequencies."
        )
    )
    ap.add_argument("--snp-freqs",       required=True)
    ap.add_argument("--features",        required=True)
    ap.add_argument("--hap-names",       dest="hap_names", required=True)
    ap.add_argument("--model",           required=True)
    ap.add_argument("--mid-model",       dest="mid_model", default=None)
    ap.add_argument("--mid-switch",      dest="mid_switch", type=float, default=None)
    ap.add_argument("--low-model",       dest="low_model", default=None)
    ap.add_argument("--low-switch",      dest="low_switch", type=float, default=None)
    ap.add_argument("--sims",            required=True)
    ap.add_argument("--regions",         required=True)
    ap.add_argument("--output",          required=True)
    ap.add_argument("--error-threshold", dest="error_threshold", type=float, required=True)
    ap.add_argument("--coarse-sizes",    dest="coarse_sizes", default="30001,50001,70001,90001,120001,150001,200001,250001")
    ap.add_argument("--refine-step",     dest="refine_step", type=int, default=5000)
    ap.add_argument("--step", "--step-start", dest="step", type=int, default=20000)
    ap.add_argument("--min-snps",        dest="min_snps", type=int, default=10)
    ap.add_argument("--max-snps",        dest="max_snps", type=int, default=400)
    ap.add_argument("--burnin",          dest="burnin", type=int, default=80)
    ap.add_argument("--sampling",        dest="sampling", type=int, default=20)
    ap.add_argument("--strict",          nargs='?', dest="strict", type=float, const=math.nan, default=None)
    ap.add_argument("--strict-step",     dest="strict_step", type=int, default=None)

    args = ap.parse_args()

    if args.strict is not None:
        strict_cutoff = args.error_threshold if math.isnan(args.strict) else args.strict
    else:
        strict_cutoff = None
    strict_step = args.strict_step or args.step
    coarse_sizes = [int(x) for x in args.coarse_sizes.split(",")]

    hap_cols   = read_list_file(args.hap_names)
    fieldnames = ["chrom","sim","pos","window_bp","Predicted_Error"] + hap_cols

    global output_fh, csv_writer
    out_path = Path(args.output)
    if out_path.exists() and out_path.stat().st_size > 0:
        resume_df = pd.read_csv(args.output)
        print(f"Resuming from existing output ({len(resume_df)} rows)")
        output_fh = open(args.output, "a", newline="")
        csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
    else:
        resume_df = None
        output_fh = open(args.output, "w", newline="")
        csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
        csv_writer.writeheader()

    # load base model + scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device, weights_only=False)
    n_haps = len(hap_cols)
    n_tab  = len(ckpt["scaler_mean"])
    model  = Predictor(n_haps, n_tab).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_             = ckpt["scaler_mean"]
    scaler.scale_            = ckpt["scaler_scale"]
    scaler.feature_names_in_ = np.array(read_list_file(args.features))

    mid_model = None
    if args.mid_model:
        ckpt_mid = torch.load(args.mid_model, map_location=device, weights_only=False)
        mid_model = Predictor(n_haps, n_tab).to(device)
        mid_model.load_state_dict(ckpt_mid["model_state_dict"])
        mid_model.eval()

    low_model = None
    if args.low_model:
        ckpt_low = torch.load(args.low_model, map_location=device, weights_only=False)
        low_model = Predictor(n_haps, n_tab).to(device)
        low_model.load_state_dict(ckpt_low["model_state_dict"])
        low_model.eval()

    sims    = read_list_file(args.sims)
    regions = read_regions_file(args.regions)

    for chrom, rstart, rend in regions:
        df_all = pd.read_csv(args.snp_freqs).query(
            "chrom==@chrom and pos>=@rstart and pos<=@rend"
        ).reset_index(drop=True)
        if df_all.empty:
            continue
        for sim in sims:
            if df_all.empty:
                continue

            if resume_df is not None:
                mask = (resume_df["chrom"] == chrom) & (resume_df["sim"] == sim)
                if mask.any():
                    last_pos = int(resume_df.loc[mask, "pos"].max())
                    print(f"Resuming {chrom}:{rstart}-{rend}, sim={sim} from pos > {last_pos}")
                    df_sub = df_all[df_all.pos > last_pos].reset_index(drop=True)
                else:
                    df_sub = df_all.copy()
            else:
                df_sub = df_all.copy()

            if not df_sub.empty:
                adaptive_windowing(
                    df_sub, sim, None,
                    model, scaler, args.error_threshold,
                    hap_cols, coarse_sizes, args.refine_step,
                    args.min_snps, args.max_snps, args.step,
                    args.burnin, args.sampling,
                    mid_model, args.mid_switch,
                    low_model, args.low_switch,
                    strict_cutoff, strict_step
                )

    output_fh.close()

if __name__ == "__main__":
    main()
