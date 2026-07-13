"""
Cross-config ensemble test script.
Loads models from multiple configs/seeds, runs fresh sequential inference
(single GPU, no DDP shuffling), then computes full + OOD metrics.
"""
from models import CatESO
from utils import set_seed, graph_collate_func
from configs import get_cfg_defaults
from dataloader import ESIDataset
from torch.utils.data import DataLoader
from run.tester_reg import Tester_Reg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch, os, warnings, argparse
import pandas as pd
import numpy as np
from time import time

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_ood_indices(train_clusters, test_clusters):
    """Adapt from CatPred"""
    return [i for i, c in enumerate(test_clusters) if c not in train_clusters]


def print_metrics(label, y_label, y_pred):
    pcc, _ = pearsonr(y_label, y_pred)
    rmse = np.sqrt(mean_squared_error(y_label, y_pred))
    mae = mean_absolute_error(y_label, y_pred)
    r2 = r2_score(y_label, y_pred)
    print(f"======> {label}")
    print(f"Sample Num: {len(y_label)}")
    print(f"PCC:  {pcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")


# ── Config: each (model_yaml, data_yaml, weight_folder, seeds) is one group ──
CONFIGS = [
    (
        "configs/model1/model.yaml",
        "configs/model1/data.yaml",
        "./results/model1",
        [10,20,30,40,50],
    ),
    (
        "configs/model2/model.yaml",
        "configs/model2/data.yaml",
        "./results/model2",
        [10,20,30,40,50],
    ),
]
# ─────────────────────────────────────────────────────────────────────────────


def build_test_loader(data_yaml, split="test"):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(data_yaml)
    data_folder = cfg.SOLVER.DATA
    test_path = os.path.join(data_folder, f"{split}.csv")
    df_test = pd.read_csv(test_path)
    dataset = ESIDataset(df_test.index.values, df_test, "regression")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=graph_collate_func,
    )
    return loader, df_test, os.path.join(data_folder, "train.csv")


def main():
    parser = argparse.ArgumentParser(description="Cross-config ensemble test")
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Comma-separated list of seeds to use for ALL configs, e.g. --seed 10,20",
    )
    args = parser.parse_args()

    # Parse seeds from CLI; if not provided, keep each config's original seeds
    if args.seed is not None:
        cli_seeds = [int(s.strip()) for s in args.seed.split(",")]
        configs = [(m, d, w, cli_seeds) for m, d, w, _ in CONFIGS]
    else:
        configs = CONFIGS

    set_seed(42)
    # ── Use the first config's data_yaml to resolve the test/train paths ─────
    _, data_yaml_0, _, _ = configs[0]
    test_loader, df_test, train_csv_path = build_test_loader(data_yaml_0)
    df_train = pd.read_csv(train_csv_path)
    # ─────────────────────────────────────────────────────────────────────────

    y_pred_all = []
    y_label = None
    total_models = 0

    for model_yaml, data_yaml, weight_folder, seeds in configs:
        cfg = get_cfg_defaults()
        cfg.merge_from_file(model_yaml)
        cfg.merge_from_file(data_yaml)
        model_name = os.path.splitext(os.path.basename(model_yaml))[0]

        for seed in seeds:
            weight_path = os.path.join(weight_folder, f"{model_name}_{seed}", "best_model_epoch.pth")
            if not os.path.exists(weight_path):
                print(f"[SKIP] {weight_path} not found")
                continue

            model = CatESO(**cfg)
            tester = Tester_Reg(model, device, test_loader, weight_path, **cfg)
            _, y_pred, y_lbl = tester.test()

            y_pred_all.append(np.array(y_pred).flatten())
            if y_label is None:
                y_label = np.array(y_lbl).flatten()
            total_models += 1
            print(f"  Loaded: {os.path.basename(weight_folder)}/seed={seed}")

    print(f"\nTotal models in ensemble: {total_models}\n")

    y_pred_ens = np.mean(y_pred_all, axis=0)

    # ── Full-set metrics ─────────────────────────────────────────────────────
    print_metrics("For Threshold 100 (All samples)", y_label, y_pred_ens)

    # ── OOD-subset metrics ───────────────────────────────────────────────────
    for N in [99, 80, 60, 40]:
        col = f"sequence_{N}cluster"
        train_clusters = set(df_train[col])
        ood_idx = get_ood_indices(train_clusters, df_test[col])
        print_metrics(
            f"For Threshold {N} (OOD, n={len(ood_idx)})",
            y_label[ood_idx],
            y_pred_ens[ood_idx],
        )


if __name__ == "__main__":
    s = time()
    main()
    print(f"\nTotal running time: {round(time()-s, 2)}s")
