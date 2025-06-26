#!/usr/bin/env python3

"""
Model validation, outs error predictions on simluated populations and verbose logs on window search process.
"""
import argparse
import logging
import sys
import re
import csv
from pathlib import Path
import threading
import math
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

#set up output/log files
output_fh = None
csv_writer = None
write_lock = threading.Lock()

# ─────────────────────────── helper functions ──────────────────────────── #

#set up logging
def setup_logging(log_file: str) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    return logging.getLogger(__name__)

def read_list_file(path: str) -> list[str]:
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

#parse regions file
def read_regions_file(path: str) -> list[tuple[str,int,int]]:
    regions = []
    pat = re.compile(r'^([^:]+):(\d+):(\d+)$')
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
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


# ─────────────────────────── feature calculation ──────────────────────────── #

#haplotype frequency model
def mcmc_model(X, b):
    p = numpyro.sample("p", dist.Dirichlet(jnp.ones(X.shape[1])))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.2))
    mu = jnp.dot(X, p)
    numpyro.sample("y_obs", dist.Normal(mu, sigma), obs=b)

#haplotype frequency inference
def mcmc_haplotype_freq(X: np.ndarray, b: np.ndarray,
                        num_warmup: int=80, num_samples: int=20,
                        num_chains: int=1, rng_seed: int=0
) -> tuple[np.ndarray, dict[str,float]]:
    kernel = NUTS(mcmc_model)
    mcmc = MCMC(kernel, num_warmup=num_warmup,
                num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(rng_seed), X, b,
             extra_fields=("diverging","num_steps"))
    samples = mcmc.get_samples(group_by_chain=True)
    extra   = mcmc.get_extra_fields(group_by_chain=True)
    p_samps = jnp.reshape(samples["p"],(-1,X.shape[1]))
    p_np    = np.array(p_samps)
    divr    = float(jnp.mean(extra["diverging"]))
    steps   = extra["num_steps"]
    depth   = int((jnp.log2(steps+1e-8).astype(int)+1).max())
    return p_np, {"divergence_rate":divr, "max_tree_depth":depth}

#least-squares estimator for features
def lsei_haplotype_estimator(X: np.ndarray, b: np.ndarray,
                             lb: float=0.0) -> np.ndarray:
    k = X.shape[1]
    cons   = ({'type':'eq','fun':lambda p: p.sum()-1.0},)
    bounds = [(lb,1.0)]*k
    p0     = np.ones(k)/k
    def obj(p): return ((X.dot(p)-b)**2).sum()
    res = minimize(obj,p0,method='SLSQP',
                   bounds=bounds,constraints=cons)
    if not res.success:
        bounds = [(0.0,1.0)]*k
        res = minimize(obj,p0,method='SLSQP',
                       bounds=bounds,constraints=cons)
    return res.x if res.success else np.full(k,np.nan)

#average snp frequency per haplotype
def compute_avg_snpfreqs(df: pd.DataFrame, sim: str) -> np.ndarray:
    n_haps = sum(1 for c in df.columns if c.startswith("B"))
    out    = np.full(n_haps, np.nan)
    vals   = df[sim].astype(float)
    for i in range(n_haps):
        mask = (df[f"B{i+1}"]==1)
        if mask.any():
            out[i] = vals[mask].mean()
    return out

#effective rank of each window
def compute_effective_rank(X: np.ndarray) -> float:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_norm   = s/(s.sum()+1e-8)
    ent      = -np.sum(s_norm*np.log(s_norm+1e-8))
    return float(np.exp(ent))

#clean/clip features
def clean_features(feats: dict[str, float],
                   clip_min: float = -1e6,
                   clip_max: float = 1e6) -> dict[str, float]:
    for k, v in feats.items():
        if k in ("start", "end"):
            continue
        if isinstance(v, (int, float, np.number)):
            vv = np.nan_to_num(v,
                               nan=0.0,
                               posinf=clip_max,
                               neginf=clip_min)
            feats[k] = float(np.clip(vv, clip_min, clip_max))
    return feats

#build features row for window
def extract_window_features(
    window_df: pd.DataFrame,
    sim: str,
    true_df: pd.DataFrame|None,
    hap_cols: list[str],
    num_warmup: int,
    num_samples: int
) -> dict[str,float]|None:
    n = len(window_df)
    if n == 0:
        return None
    start_c = float(window_df.pos.min())
    end_c   = float(window_df.pos.max())
    window_bp = end_c - start_c + 1
    b_vals   = window_df[sim].astype(float).values
    mean_f,std_f = b_vals.mean(), b_vals.std()
    snr      = mean_f/std_f if std_f>0 else np.nan
    X        = window_df[hap_cols].astype(float).values
    cond_num = np.linalg.cond(X) if X.shape[0]>=X.shape[1] else np.inf
    min_col_dist = (
        min(np.sum(X[:,i]!=X[:,j])
            for i in range(X.shape[1]) for j in range(i+1,X.shape[1]))
        if X.shape[1]>1 else 0
    )
    row_sums = X.sum(axis=1)
    avg_row, row_std = row_sums.mean(), row_sums.std()
    e_rank   = compute_effective_rank(X)

    # use provided warmup & sample counts
    p_samps, diag = mcmc_haplotype_freq(
        X, b_vals,
        num_warmup=num_warmup,
        num_samples=num_samples
    )

    avg_unc = float(np.std(p_samps,axis=0).mean())
    avg_skw = float(skew(p_samps,axis=0).mean())
    avg_krt = float(kurtosis(p_samps,axis=0).mean())
    sig     = float(std_f)+1e-8
    H       = (X.T @ X) / (sig**2 + 1e-8)
    eigs    = np.linalg.eigvals(H)
    imp_grad= float(np.log(np.median(1.0/(np.abs(eigs)+1e-8))+1e-8))
    lsei    = lsei_haplotype_estimator(X, b_vals)
    avgf    = compute_avg_snpfreqs(window_df, sim)

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
    if true_df is not None:
        mid = (start_c + end_c) / 2
        idx = (true_df.pos - mid).abs().idxmin()
        true_row = true_df.loc[idx, hap_cols].astype(float).values
        feats["error"] = float(np.abs(p_samps.mean(axis=0) - true_row).sum())

    return clean_features(feats)

# ───────────────────────── model ────────────────────────────
class SNPTransformerEncoder(nn.Module):
    def __init__(self, n_haps, embed_dim=64, heads=4, layers=3):
        super().__init__()
        self.proj = nn.Linear(n_haps, embed_dim)
        blk      = nn.TransformerEncoderLayer(embed_dim, heads, batch_first=True)
        self.enc = nn.TransformerEncoder(blk, layers)
        self.pool= nn.Linear(embed_dim, 1)

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
                nn.Dropout(drop)
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

# ───────────────────────── adaptive windower ────────────────────────────
def adaptive_windowing(
    df, sim, region, true_df,
    model, scaler, err_thr,
    hap_cols, coarse_sizes, step_refine,
    min_snps, max_snps, step_start,
    num_warmup, num_samples,
    mid_model=None, mid_switch=None,
    low_model=None, low_switch=None,
    strict_cutoff=None, strict_step=None
):
    global output_fh, csv_writer

    logger = logging.getLogger(__name__)
    cols   = scaler.feature_names_in_
    start, end = int(df.pos.min()), int(df.pos.max())

    logger.info(f"START adaptive windowing: sim={sim}, region={region}, threshold={err_thr:.3f}")
    logger.info(f"Coarse sizes: {coarse_sizes}, refine step: {step_refine}bp")

    while start < end:
        best, found = None, False
        # check if strict mode
        curr_thr = strict_cutoff if strict_cutoff is not None else err_thr

        # coarse scan
        for cs in coarse_sizes:
            we  = start + cs
            win = df[(df.pos >= start) & (df.pos < we)]
            if len(win) < min_snps:
                continue

            feats = extract_window_features(
                win, sim, true_df, hap_cols,
                num_warmup, num_samples
            )
            if feats is None:
                continue
            meta = pd.DataFrame([feats])[cols].replace([np.inf, -np.inf], np.nan)
            if meta.isna().any(axis=1).item():
                continue

            # base-model prediction
            X_tab = torch.tensor(scaler.transform(meta), dtype=torch.float32)
            M     = win[hap_cols].astype(float).values
            if len(M) > max_snps:
                M = M[-max_snps:]
            else:
                M = np.vstack([np.zeros((max_snps - len(M), M.shape[1])), M])
            Xr = torch.tensor(M.astype(np.float32)).unsqueeze(0)

            with torch.no_grad():
                pe_base = float(model(Xr, X_tab).item())

            # step to mid and low models if provided
            pe = pe_base
            log_parts = [f"Pred_base={pe_base:.4f}"]
            if mid_model is not None and mid_switch is not None and pe_base <= mid_switch:
                with torch.no_grad():
                    pe_mid = float(mid_model(Xr, X_tab).item())
                log_parts.append(f"Pred_mid={pe_mid:.4f}")
                pe = pe_mid
                if low_model is not None and low_switch is not None and pe_mid <= low_switch:
                    with torch.no_grad():
                        pe_low = float(low_model(Xr, X_tab).item())
                    log_parts.append(f"Pred_low={pe_low:.4f}")
                    pe = pe_low

            logger.info(f"  Window {start}-{we} ({cs}bp): " +
                        ", ".join(log_parts) +
                        f", True={feats.get('error', np.nan):.4f}")

            te = feats.get("error", np.nan)

            if best is None or pe < best["pe"]:
                best = {"pe": pe, "we": we, "cs": cs, "te": te, "feats": feats, "win": win}
            if pe <= curr_thr:
                logger.info(f"    coarse hit @ size {cs}bp: {pe:.4f} ≤ {curr_thr:.4f}")
                found = True
                break

        # strict‐mode
        if not found and strict_cutoff is not None:
            logger.info(f"No window ≤ {curr_thr:.4f} at position {start}, skipping due to strict mode")
            start += strict_step
            continue

        if best is None:
            logger.info("No candidate found; terminating.")
            start += start_step
            continue

        # refinement
        cs0  = best["cs"]
        prev = [s for s in coarse_sizes if s < cs0]
        low  = prev[-1] if prev else coarse_sizes[0]
        refine_thr = err_thr
        logger.info(f"  refine between {low}–{cs0}bp in steps of {step_refine}bp")
        for W in range(low, cs0 + 1, step_refine):
            we  = start + W
            win = df[(df.pos >= start) & (df.pos < we)]
            if len(win) < min_snps:
                continue

            feats = extract_window_features(
                win, sim, true_df, hap_cols,
                num_warmup, num_samples
            )
            if feats is None:
                continue
            meta = pd.DataFrame([feats])[cols].replace([np.inf, -np.inf], np.nan)
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
                pe_base = float(model(Xr, X_tab).item())

            # same switch logic
            pe = pe_base
            log_parts = [f"Pred_base={pe_base:.4f}"]
            if mid_model is not None and mid_switch is not None and pe_base <= mid_switch:
                with torch.no_grad():
                    pe_mid = float(mid_model(Xr, X_tab).item())
                log_parts.append(f"Pred_mid={pe_mid:.4f}")
                pe = pe_mid
                if low_model is not None and low_switch is not None and pe_mid <= low_switch:
                    with torch.no_grad():
                        pe_low = float(low_model(Xr, X_tab).item())
                    log_parts.append(f"Pred_low={pe_low:.4f}")
                    pe = pe_low

            logger.info(f"    refine {W}bp: " +
                        ", ".join(log_parts) +
                        f", True={feats.get('error', np.nan):.4f}")

            te = feats.get("error", np.nan)
            if pe < best["pe"] or pe <= refine_thr:
                logger.info(f"      refine hit @ {W}bp: {pe:.4f} ≤ {refine_thr:.4f}")
                best = {"pe": pe, "we": we, "cs": W, "te": te, "feats": feats, "win": win}
                break

        # write record 
        rec = {
            "chrom": df.chrom.iloc[0],
            "region": region,
            "sim": sim,
            "start": int(best["feats"]["start"]),
            "end":   int(best["feats"]["end"]),
            "window_bp": best["cs"],
            "Predicted_Error": best["pe"],
            "error": best["te"]
        }
        rec.update({k: v for k, v in best["feats"].items() if k not in ("start", "end", "error")})
        for h in hap_cols:
            rec[h] = float(best["win"][h].mean()) if not best["win"].empty else np.nan

        with write_lock:
            csv_writer.writerow(rec)
            output_fh.flush()

        nxt = df[df.pos >= start + step_start]
        if nxt.empty:
            break
        start = int(nxt.pos.iloc[0])

def main():
    ap = argparse.ArgumentParser(
        description=textwrap.dedent("""\
        evaluate model from haplomatic-train. optionally accepts models specialized on medium/low error in addition to base model.

        required files
        -------------
        --snp-freqs: csv of SNP frequencies from training populations with <chrom>, <pos>, <haplotypes...>, <populations...>
        --features: csv of features calculated by haplomatic-window
        --hap-names: text file of haplotype names, one per row
        --model: model from haplomatic-train
        --sims: text file of simulated populations to evaluate model on
        --regions: genomic regions to evaluate model over (contig:start:end)
        """                         
        )
    )
    ap.add_argument("--snp-freqs",       required=True)
    ap.add_argument("--features",        required=True)
    ap.add_argument("--hap-names",  required=True)
    ap.add_argument("--model",           required=True)
    ap.add_argument("--mid-model",       default=None,
                        help="path to intermediate (mid-error) model")
    ap.add_argument("--mid-switch",      type=float, default=None,
                        help="predicted error threshold to switch from base to mid-model")
    ap.add_argument("--low-model",       default=None,
                        help="path to low-error model")
    ap.add_argument("--low-switch",      type=float, default=None,
                        help="predicted error threshold to switch from mid to low-model")
    ap.add_argument("--sims",            required=True)
    ap.add_argument("--regions",         required=True)
    ap.add_argument("--true-freq-dir",   default=None)
    ap.add_argument("--output",          required=True)
    ap.add_argument("--log-file",        default="adaptive_windowing.log")
    ap.add_argument("--error-threshold", type=float, required=True)
    ap.add_argument("--coarse-sizes",    default="30001,50001,70001,90001,120001,150001,200001,250001",
                        help="comma-separated coarse window sizes in bp")
    ap.add_argument("--refine-step",     type=int, default=5000)
    ap.add_argument("--step-start", "--step", type=int, default=20000,
                        help="advance window by this many bp after recording")
    ap.add_argument("--min-snps",        type=int, default=10,
                        help="minimum SNPs per window")
    ap.add_argument("--max-snps",        type=int, default=400,
                        help="maximum SNPs per window for model input")
    ap.add_argument("--burnin",          type=int, default=80,
                        help="number of MCMC warmup iterations")
    ap.add_argument("--sampling",        type=int, default=20,
                        help="number of MCMC samples after warmup")
    ap.add_argument("--strict",          nargs='?', type=float, const=math.nan, default=None,
                        help="enable strict mode: only accept windows with pred_error ≤ this cutoff; if no value given, uses --error-threshold")
    ap.add_argument("--strict-step",     type=int, default=None,
                        help="bp to advance when skipping in strict mode (defaults to --step-start)")

    args = ap.parse_args()


    if args.strict is not None:
        if math.isnan(args.strict):
            strict_cutoff = args.error_threshold
        else:
            strict_cutoff = args.strict
    else:
        strict_cutoff = None

    if args.strict_step is None:
        strict_step = args.step_start
    else:
        strict_step = args.strict_step

    coarse_sizes = [int(x) for x in args.coarse_sizes.split(",")]

    static_cols  = ["chrom","region","sim","start","end","window_bp","Predicted_Error","error"]
    raw_features = read_list_file(args.features)
    hap_cols     = read_list_file(args.hap_names)
    feature_cols = [f for f in raw_features if f not in static_cols and f not in hap_cols]
    fieldnames   = static_cols + feature_cols + hap_cols

    logger = setup_logging(args.log_file)

    global output_fh, csv_writer
    out_path = Path(args.output)
    if out_path.exists() and out_path.stat().st_size > 0:
        resume_df = pd.read_csv(args.output)
        print(f"Resuming from existing output ({len(resume_df)} rows)")
        logger.info(f"Resuming from existing output ({len(resume_df)} rows)")
        output_fh = open(args.output, "a", newline="")
        csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames, extrasaction='ignore')
    else:
        resume_df = None
        output_fh = open(args.output, "w", newline="")
        csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames, extrasaction='ignore')
        csv_writer.writeheader()

    # load base model + scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(args.model, map_location=device, weights_only=False)
    n_haps = len(read_list_file(args.hap_names))
    n_tab  = len(ckpt["scaler_mean"])
    model  = Predictor(n_haps, n_tab).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scaler = StandardScaler()
    scaler.mean_             = ckpt["scaler_mean"]
    scaler.scale_            = ckpt["scaler_scale"]
    scaler.feature_names_in_ = np.array(read_list_file(args.features))

    # load mid-model
    if args.mid_model:
        ckpt_mid  = torch.load(args.mid_model, map_location=device, weights_only=False)
        mid_model = Predictor(n_haps, n_tab).to(device)
        mid_model.load_state_dict(ckpt_mid["model_state_dict"])
        mid_model.eval()
    else:
        mid_model = None

    # load low-model
    if args.low_model:
        ckpt_low  = torch.load(args.low_model, map_location=device, weights_only=False)
        low_model = Predictor(n_haps, n_tab).to(device)
        low_model.load_state_dict(ckpt_low["model_state_dict"])
        low_model.eval()
    else:
        low_model = None

    sims    = read_list_file(args.sims)
    regions = read_regions_file(args.regions)

    for chrom, rstart, rend in regions:
        df_all = pd.read_csv(args.snp_freqs).query(
            "chrom==@chrom and pos>=@rstart and pos<=@rend"
        ).reset_index(drop=True)
        if df_all.empty:
            logger.warning(f"No SNPs in {chrom}:{rstart}-{rend}")
            continue
        region_str = f"{chrom}:{rstart}-{rend}"
        for sim in sims:
            if resume_df is not None:
                mask = (resume_df["region"]==region_str)&(resume_df["sim"]==sim)
                if mask.any():
                    last_start = int(resume_df.loc[mask,"start"].max())
                    resume_pos = last_start + args.step_start
                    print(f"Resuming {region_str}, sim={sim} from pos > {resume_pos}")
                    logger.info(f"Resuming {region_str}, sim={sim} from pos > {resume_pos}")
                    subset = df_all[df_all.pos>resume_pos].reset_index(drop=True)
                else:
                    subset = df_all.copy()
            else:
                subset = df_all.copy()

            if not subset.empty:
                true_df = None
                if args.true_freq_dir:
                    tf = Path(args.true_freq_dir)/f"{sim}_true_freqs.csv"
                    if tf.exists():
                        true_df = pd.read_csv(tf)

                adaptive_windowing(
                    subset, sim, region_str, true_df,
                    model, scaler,
                    args.error_threshold,
                    hap_cols,
                    coarse_sizes,
                    args.refine_step,
                    args.min_snps,
                    args.max_snps,
                    args.step_start,
                    args.burnin,
                    args.sampling,
                    mid_model, args.mid_switch,
                    low_model, args.low_switch,
                    strict_cutoff, strict_step
                )

    output_fh.close()
    logger.info("model eva complete")

if __name__=="__main__":
    main()
