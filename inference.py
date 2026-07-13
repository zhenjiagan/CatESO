"""
Ensemble inference: automatically load **all** model_{N}_M{k}.pth weights under
ensemble_ckpt/, predict row-by-row for --split_csv, average every model's
prediction in log10(kcat) space, and write --out_csv (all original columns +
pred_log10_kcat).

Weight -> model-config mapping (by the M{k} in the filename):
    model_{seed}_M1.pth  ->  configs/model1/{model,data}.yaml
    model_{seed}_M2.pth  ->  configs/model2/{model,data}.yaml
    general rule: M{k}  ->  configs/model{k}

Usage:
    python inference.py --split_csv in.csv --out_csv out.csv
    # scans ./ensemble_ckpt by default; use --ckpt_dir to point elsewhere
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "run")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run.configs import get_cfg_defaults
from run.dataloader import ESIDataset
from run.utils import graph_collate_func, set_seed
from models import CatESO

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "ensemble_ckpt")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
_CKPT_RE = re.compile(r"^model_(\d+)_(M\d+)\.pth$")


def _build_config_dict(cfg):
    """Build the config dict passed to CatESO from a yacs CfgNode (matches main_optimize.py)."""
    return {
        "DRUG":    dict(cfg.DRUG),
        "PROTEIN": dict(cfg.PROTEIN),
        "DECODER": dict(cfg.DECODER),
        "ICFE":    dict(cfg.ICFE),
        "MHSA":    dict(cfg.MHSA),
        "MCDC":    dict(cfg.MCDC),
        "FUSION":  dict(cfg.FUSION),
    }


def discover_checkpoints(ckpt_dir):
    """Scan ckpt_dir and return [(path, model_key), ...] sorted by (M{k}, seed)."""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"ensemble_ckpt directory not found: {ckpt_dir}")
    found = []
    for fname in os.listdir(ckpt_dir):
        m = _CKPT_RE.match(fname)
        if m:
            found.append((m.group(2), int(m.group(1)), os.path.join(ckpt_dir, fname)))
    if not found:
        raise FileNotFoundError(
            f"No weight files matching model_{{N}}_M{{k}}.pth found in {ckpt_dir}"
        )
    found.sort(key=lambda x: (x[0], x[1]))
    return [(path, key) for key, _, path in found]


def config_dir_for_key(model_key):
    """'M1' -> configs/model1, 'M2' -> configs/model2 ..."""
    cfg_dir = os.path.join(CONFIGS_DIR, f"model{model_key[1:]}")
    if not os.path.isdir(cfg_dir):
        raise FileNotFoundError(
            f"config directory for {model_key} not found: {cfg_dir}\n"
            f"expected {cfg_dir}/model.yaml (data.yaml optional)."
        )
    return cfg_dir


def load_model(ckpt_path, model_key):
    """Load the config + weights for model_key and return the CatESO model in eval mode."""
    cfg = get_cfg_defaults()
    cfg_dir = config_dir_for_key(model_key)
    model_yaml = os.path.join(cfg_dir, "model.yaml")
    data_yaml = os.path.join(cfg_dir, "data.yaml")
    if not os.path.isfile(model_yaml):
        raise FileNotFoundError(f"model config not found: {model_yaml}")
    cfg.merge_from_file(model_yaml)
    if os.path.isfile(data_yaml):
        cfg.merge_from_file(data_yaml)

    model = CatESO(**_build_config_dict(cfg))
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def build_loader(csv_path, batch_size):
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"split_csv not found: {csv_path}")
    df = pd.read_csv(csv_path).reset_index(drop=True)
    for col in ("SMILES", "Protein_Path"):
        if col not in df.columns:
            raise KeyError(f"input CSV missing required column: {col}")
    # For label-free inference, add a placeholder Score column for ESIDataset (unused in prediction)
    if "Score" not in df.columns:
        df["Score"] = 0.0
    dataset = ESIDataset(df.index.values, df, "regression")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=graph_collate_func,
    )
    return loader, df, csv_path


@torch.no_grad()
def predict(model, loader):
    """Run one forward pass over the whole loader and return a 1D prediction array."""
    preds = []
    for v_d, v_p, _labels, v_d_mask, v_p_mask in loader:
        v_d, v_p = v_d.to(device), v_p.to(device)
        v_d_mask, v_p_mask = v_d_mask.to(device), v_p_mask.to(device)
        _, _, _, score = model(v_d, v_p, v_d_mask, v_p_mask)
        preds.append(np.asarray(score.squeeze(-1).cpu()).reshape(-1))
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser(
        description="Ensemble inference: load all weights under ensemble_ckpt/ -> pred_log10_kcat"
    )
    p.add_argument("--split_csv", required=True,
                   help="input CSV (must contain SMILES, Protein_Path; row order = output order)")
    p.add_argument("--out_csv", required=True,
                   help="output CSV: all original columns + pred_log10_kcat")
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR,
                   help=f"ensemble weight directory (default: {DEFAULT_CKPT_DIR})")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    set_seed(42)

    checkpoints = discover_checkpoints(args.ckpt_dir)
    print(f"[ensemble] found {len(checkpoints)} weights in {args.ckpt_dir}:")
    for path, key in checkpoints:
        print(f"    {os.path.basename(path):<24} {key} -> configs/model{key[1:]}")

    test_loader, df_in, in_path = build_loader(args.split_csv, args.batch_size)

    y_pred_all: list[np.ndarray] = []
    for i, (ckpt_path, model_key) in enumerate(checkpoints, 1):
        model = load_model(ckpt_path, model_key)
        y_pred = predict(model, test_loader)
        if len(y_pred) != len(df_in):
            raise RuntimeError(
                f"{os.path.basename(ckpt_path)}: #predictions {len(y_pred)} != #rows {len(df_in)}"
            )
        y_pred_all.append(y_pred)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{i}/{len(checkpoints)}] {os.path.basename(ckpt_path)} done")

    y_mean = np.mean(np.stack(y_pred_all, axis=0), axis=0)

    out_path = os.path.abspath(args.out_csv)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out = df_in.copy()
    df_out["pred_log10_kcat"] = y_mean
    df_out.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"  ensemble: mean of {len(y_pred_all)} models")
    print(f"  rows: {len(df_out)}  cols: {list(df_out.columns)}")
    print(f"  input: {in_path}")


if __name__ == "__main__":
    t0 = time()
    main()
    print(f"\nTotal running time: {round(time() - t0, 2)}s")
