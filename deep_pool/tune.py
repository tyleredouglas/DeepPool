#!/usr/bin/env python3
# deep_pool/tune.py

"""
Fine-tune pre-trained models from haplomatic-train
"""

import os
import sys
import argparse
import math
import numpy  as np
import pandas as pd
import numpy._core.multiarray as marray
from torch.serialization import add_safe_globals
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data    import Dataset, DataLoader
from   torch.cuda.amp      import autocast, GradScaler

# ─────────────────────────── helper functions ──────────────────────────── #

def read_list(txt: str):
    with open(txt) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# ───────────────────────── defining training instance (SNP windows) ────────────────────────────
class ErrorDataset(Dataset):
    """Return (X_raw, X_tab, y) for one window."""
    def __init__(self, df_win: pd.DataFrame, snp_df: pd.DataFrame,
                 idx: np.ndarray, X_tab: np.ndarray, y: np.ndarray,
                 max_snps: int, hap_cols: list[str]):

        self.df   = df_win.loc[idx].reset_index(drop=True)
        self.snp  = snp_df.sort_values('pos').reset_index(drop=True)
        self.pos  = self.snp['pos'].values
        self.haps = hap_cols
        self.max  = max_snps
        self.Xtab = torch.from_numpy(X_tab[idx]).float()
        self.y    = torch.from_numpy(y[idx]).float()

        self.ranges = []
        for _, row in self.df.iterrows():
            lo = np.searchsorted(self.pos, row.start, side='left')
            hi = np.searchsorted(self.pos, row.end,   side='left')
            self.ranges.append((lo, hi))

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        lo, hi = self.ranges[i]
        sub    = self.snp.iloc[lo:hi][self.haps].astype(float).values
        n, k   = sub.shape
        if n >= self.max:
            mat = sub[-self.max:]
        else:
            pad = np.zeros((self.max-n, k), dtype=np.float32)
            mat = np.vstack([pad, sub])

        return (
            torch.from_numpy(mat.astype(np.float32)),
            self.Xtab[i],
            self.y[i]
        )


# ───────────────────────── model ────────────────────────────
class SNPTransformerEncoder(nn.Module):
    def __init__(self, n_haps:int, embed:int=64, heads:int=4, layers:int=3):
        super().__init__()
        self.proj  = nn.Linear(n_haps, embed)
        block      = nn.TransformerEncoderLayer(embed, heads, batch_first=True)
        self.enc   = nn.TransformerEncoder(block, layers)
        self.pool  = nn.Linear(embed, 1)

    def forward(self, x):
        z = self.proj(x)
        z = self.enc(z)
        w = F.softmax(self.pool(z), dim=1)
        return (w * z).sum(dim=1)


class TabularRegressor(nn.Module):
    def __init__(self, inp:int, hidden=(256,128), drop=0.3):
        super().__init__()
        layers, dims = [], [inp] + list(hidden)
        for i in range(len(hidden)):
            layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.Dropout(drop)
            ]
        layers.append(nn.Linear(dims[-1],1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(1)


class Predictor(nn.Module):
    def __init__(self, n_haps:int, n_tab:int, embed:int=64, dropout:float=0.3):
        super().__init__()
        self.enc = SNPTransformerEncoder(n_haps, embed)
        self.reg = TabularRegressor(embed + n_tab, drop=dropout)

    def forward(self, Xraw, Xtab):
        z = self.enc(Xraw)
        return self.reg(torch.cat([z, Xtab], dim=1))


# ──────────────────────── training  ─────────────────────────
def train(model, train_loader, val_loader,
          opt, sched, scaler_grad, device,
          start_epoch, n_epochs, ckpt_path, best_ckpt_path,
          scaler_feat, best_r2, history,
          save_best, region_label="ALL"):

    print(f"[tune] starting fine-tuning: "
          f"{len(train_loader.dataset):,} windows "
          f"(val {len(val_loader.dataset):,}) | epochs {n_epochs}",
          flush=True)

    for epoch in range(start_epoch, n_epochs+1):

        # training pass 
        model.train()
        tot = 0.0
        for Xr,Xt,y in train_loader:
            Xr,Xt,y = Xr.to(device), Xt.to(device), y.to(device)
            opt.zero_grad()
            with autocast():
                pred = model(Xr,Xt)
                loss = F.mse_loss(pred, y)
            scaler_grad.scale(loss).backward()
            scaler_grad.step(opt)
            scaler_grad.update()
            tot += loss.item() * Xr.size(0)
        train_mse = tot / len(train_loader.dataset)

        # validation pass 
        model.eval()
        vtot, preds, trues = 0.0, [], []
        with torch.no_grad():
            for Xr,Xt,y in val_loader:
                Xr,Xt,y = Xr.to(device), Xt.to(device), y.to(device)
                with autocast():
                    pred = model(Xr,Xt)
                vtot += F.mse_loss(pred, y).item() * Xr.size(0)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
        val_mse = vtot / len(val_loader.dataset)

        # compute r^2
        y_true = np.concatenate(trues)
        y_pred = np.concatenate(preds)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if not finite.all():
            print(f"[tune] ⚠️  dropping {len(finite)-finite.sum()} NaN/Inf preds",
                  file=sys.stderr)
            y_true = y_true[finite]
            y_pred = y_pred[finite]

        val_r2  = r2_score(y_true, y_pred)
        sched.step(val_mse)

        # save best yet model
        if save_best and val_r2 > best_r2:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "opt_state_dict":   opt.state_dict(),
                "gscaler":          scaler_grad.state_dict(),
                "scaler_mean":      scaler_feat.mean_,
                "scaler_scale":     scaler_feat.scale_,
                "best_r2":          val_r2,
                "history":          history + [(epoch, train_mse, val_mse, val_r2)]
            }, best_ckpt_path)
            print(f"[tune] new best R²={val_r2:.3f} → {best_ckpt_path}")

        # save checkpoint
        history.append((epoch, train_mse, val_mse, val_r2))
        best_r2 = max(best_r2, val_r2)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "opt_state_dict":   opt.state_dict(),
            "gscaler":          scaler_grad.state_dict(),
            "scaler_mean":      scaler_feat.mean_,
            "scaler_scale":     scaler_feat.scale_,
            "best_r2":          best_r2,
            "history":          history
        }, ckpt_path)

        print(f"[tune] {region_label} epoch {epoch:02d} "
              f"train MSE={train_mse:.4e} val MSE={val_mse:.4e} R²={val_r2:.3f}",
              flush=True)

    return best_r2, history


# ─────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=textwrap.dedent("""\
        fine-tune pre-trained error predictor from haplomatic-train. 

        required files
        -------------
        --snp-freqs-csv: csv of SNP frequencies from training populations with <chrom>, <pos>, <haplotypes...>, <populations...>
        --features-csv: csv of features calculated by haplomatic-window
        --feature-list: which features to use for training from features-csv 
        """                         
        )
    )
    ap.add_argument("--snp-freqs-csv",      required=True)
    ap.add_argument("--features-csv",       required=True)
    ap.add_argument("--feature-list",       required=True)
    ap.add_argument("--hap-names-file",     required=True)
    ap.add_argument("--region",             default=None)
    ap.add_argument("--max-snps",           type=int,   default=400)
    ap.add_argument("--batch",              type=int,   default=64)
    ap.add_argument("--val-batch",          type=int,   default=128)
    ap.add_argument("--epochs",             type=int,   default=20,
                    help="number of fine-tuning epochs (default 20)")
    ap.add_argument("--lr",                 type=float, default=1e-4,
                    help="learning rate for fine-tuning (default 1e-4)")
    ap.add_argument("--weight-decay",       type=float, default=1e-5)
    ap.add_argument("--workers",            type=int,   default=4)
    ap.add_argument("--dropout",            type=float, default=0.3)
    ap.add_argument("--pretrained-ckpt",    required=True,
                    help="path to pre-trained checkpoint (.pt)")
    ap.add_argument("--model-name",         default=None,
                    help="basename for new fine-tuned checkpoints")
    ap.add_argument("--save-best",          action="store_true",
                    help="also keep best model")
    args = ap.parse_args()

    # device (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[tune] device = {device}", file=sys.stderr)

    # read column lists
    feat_cols = read_list(args.feature_list)
    hap_cols  = read_list(args.hap_names_file)

    # load feature table
    feat_df = pd.read_csv(args.features_csv)
    if args.region:
        feat_df = feat_df.query("chrom == @args.region").reset_index(drop=True)

    # clean data
    X_tab_df = feat_df[feat_cols].replace([np.inf, -np.inf], np.nan)
    before   = len(feat_df)
    mask     = ~X_tab_df.isna().any(axis=1)
    feat_df  = feat_df.loc[mask].reset_index(drop=True)
    X_tab    = X_tab_df.loc[mask].values
    after    = len(feat_df)
    print(f"[tune] dropped {before-after} rows with NaN/Inf", file=sys.stderr)

    # target values
    y_all = feat_df["error"].clip(lower=1e-8).values

    # load SNP freqs
    snp_df = pd.read_csv(args.snp_freqs_csv)
    if args.region:
        snp_df = snp_df.query("chrom == @args.region").reset_index(drop=True)

    # initialize model & scaler from pretrained-ckpt
    region_label = args.region or "ALL"
    model_name   = args.model_name or f"{region_label}_finetuned"
    ckpt_path    = f"{model_name}.pt"
    best_ckpt    = f"{model_name}_best.pt"

    model = Predictor(len(hap_cols), X_tab.shape[1],
                      dropout=args.dropout).to(device)

    add_safe_globals([marray._reconstruct])
    ck_pre = torch.load(args.pretrained_ckpt,
                        map_location=device,
                        weights_only=False)
    model.load_state_dict(ck_pre["model_state_dict"])
    scaler_feat = StandardScaler()
    scaler_feat.mean_, scaler_feat.scale_ = (
      ck_pre["scaler_mean"], ck_pre["scaler_scale"]
    )
    print(f"loaded pretrained weights from {args.pretrained_ckpt}",
          file=sys.stderr)

    # scale features
    X_scaled = scaler_feat.transform(X_tab)

    # split into train/val
    idx = np.arange(len(feat_df))
    np.random.seed(42); np.random.shuffle(idx)
    split = int(0.7 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]

    # build loaders
    train_ds = ErrorDataset(feat_df, snp_df, tr_idx, X_scaled, y_all,
                            args.max_snps, hap_cols)
    val_ds   = ErrorDataset(feat_df, snp_df, val_idx, X_scaled, y_all,
                            args.max_snps, hap_cols)

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True,
                              persistent_workers=(args.workers>0))
    val_loader   = DataLoader(val_ds,   batch_size=args.val_batch,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True,
                              persistent_workers=(args.workers>0))

    # new optimizer, scheduler, scaler
    opt     = optim.Adam(model.parameters(),
                         lr=args.lr,
                         weight_decay=args.weight_decay)
    sched   = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                   'min',
                                                   patience=4)
    gscaler = GradScaler()

    # resume fine-tuning if fine-tune checkpoint exists
    start_epoch, best_r2, history = 1, -math.inf, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        gscaler.load_state_dict(ck["gscaler"])
        scaler_feat.mean_, scaler_feat.scale_ = (
          ck["scaler_mean"], ck["scaler_scale"]
        )
        start_epoch = ck["epoch"] + 1
        best_r2     = ck["best_r2"]
        history     = ck["history"]
        print(f"[tune] resuming fine-tune from {ckpt_path} (epoch {start_epoch})",
              file=sys.stderr)

    # run fine-tuning
    best_r2, history = train(
        model, train_loader, val_loader,
        opt, sched, gscaler, device,
        start_epoch, args.epochs,
        ckpt_path, best_ckpt,
        scaler_feat, best_r2, history,
        save_best=args.save_best,
        region_label=region_label
    )

    # final outputs
    pd.DataFrame(history,
                 columns=["epoch","train_mse","val_mse","val_r2"]
                ).to_csv(f"history_{model_name}.csv", index=False)
    torch.save(model.state_dict(), f"{model_name}.pth")
    print("[tune] fine-tuning complete", file=sys.stderr)


if __name__ == "__main__":
    main()
