# Sequence Optimization Parameters

This document explains the important parameters used by sequence optimization
(`main_optimize.py`), corresponding to the config template
[`configs/optimization/default.yaml`](configs/optimization/default.yaml).
For how to run it and the overall workflow, see [README.md](README.md).

Optimization flow: use the trained CatESO ensemble as the objective and run
differentiable (Gumbel-Softmax) optimization of the enzyme sequence, optionally
with an ESMFold structure constraint. Each seed produces
`<OUTPUT>/optimize_output_<seed>.pth`.

## ① Input / Output (required)

| Parameter | Meaning | How to choose |
|---|---|---|
| `INIT_SEQUENCE` | Starting (wild-type) enzyme sequence | Put the enzyme sequence you want to engineer |
| `TARGET_SMILES` | Target substrate SMILES | Objective = maximize `kcat` on this substrate |
| `FIXED_INDICES` | Frozen positions (0-based) | `""`=all mutable; `"0-49"` or `"0,10,20"` locks active sites / key motifs |
| `OUTPUT` | Output directory | Each seed saves `optimize_output_<seed>.pth` |
| `SEED` | List of random seeds; **each seed is one independent run** | Give more seeds for more candidate designs (default 1–50, 50 runs) |

> `CHECKPOINT` is ignored when auto-scanning with `--ensemble_ckpt_dir`.

## ② Optimizer & iterations

| Parameter | Meaning | How to choose |
|---|---|---|
| `NUM_ITERATIONS` | Number of steps (default 800) | Larger = more thorough but slower; increase for long sequences / with structure constraint |
| `LR` | Learning rate (default 0.02, applied to soft logits) | Too large is unstable, too small is slow; 0.01–0.05 is common with `adamw` |
| `OPTIMIZER` / `OPTIMIZER_KWARGS` | Optimizer and extra kwargs | `adam/adamw/sgd/rmsprop/adagrad`; e.g. `"weight_decay=0.01"` |
| `LR_SCHEDULER` / `LR_MIN` | LR decay and lower bound | `none/linear/cosine`, decays to `LR_MIN` |
| `EARLY_STOP` / `PATIENCE` | Early stopping and patience | Enable `EARLY_STOP=True` to save compute |

## ③ Sequence parameterization: temperature annealing (Gumbel-Softmax)

| Parameter | Meaning | How to choose |
|---|---|---|
| `TEMPERATURE` | Initial temperature | High=smoother distribution, more exploration; low=closer to discrete one-hot, more "committed" |
| `ANNEAL` | Whether to anneal (temperature from `TEMPERATURE` down to `MIN_TEM`) | Enable to explore first, converge later |
| `ANNEALING_SCHEME` | Annealing curve | `fixed/linear/cosine/exp` |
| `MIN_TEM` | Final annealing temperature | — |

> Typical choices: **fixed temperature** (default) `ANNEAL=False, TEMPERATURE=MIN_TEM=1.0`, stable and reproducible;
> **annealing** with `ANNEAL=True, TEMPERATURE=2.0, MIN_TEM=0.1, cosine`, broad search early then convergence.

## ④ Straight-Through Estimator STE (align the discrete sequence with the objective)

| Parameter | Meaning | How to choose |
|---|---|---|
| `USE_STE` | Forward uses the argmax hard sequence, backward uses soft gradients | Enable `True` to make the **final discrete sequence**'s prediction more reliable |
| `STE_MODE` / `STE_INTERVAL` | STE mode / apply every N steps | `none/hard`; `None`=not periodic |
| `DECOUPLED_STE` | Decouple sampling temperature from gradient temperature (gradient uses tau=1.0) | Optional with STE |

## ⑤ Closeness to the wild type (regularization)

| Parameter | Meaning | How to choose |
|---|---|---|
| `PENALTY_TYPE` | Constrain the optimized sequence to stay close to the initial one | `kl` (distribution KL) / `l2_embedding` (embedding L2) / `""` (none) |
| `PENALTY_LAMBDA` | Penalty strength (default 0.001) | **Larger = more conservative** (fewer mutations, closer to wild type); smaller = more aggressive (more mutations) |

> Tuning `PENALTY_LAMBDA` typically controls the number of mutations, usually in the range 0.0005–0.0030. To tune it quickly, first set `USE_ESMFOLD: False`, then adjust this parameter and test a few seeds until you hit the target mutation-count range. For high-confidence sequence optimization, you can try turning off `USE_ESMFOLD: False` first and observe the optimization result to save time.

## ⑥ kcat objective

| Parameter | Meaning | How to choose |
|---|---|---|
| `KCAT_LOSS_TYPE` | Objective type | `mean_only`=maximize the ensemble mean only; `uncertainty_aware`=also penalize ensemble disagreement (more robust) |
| `KCAT_UNCERTAINTY_BETA` | Uncertainty penalty strength | Only used with `uncertainty_aware`; larger = more conservative |

## ⑦ ESMFold structure constraint (avoid generating unstable sequences)

| Parameter | Meaning | How to choose |
|---|---|---|
| `USE_ESMFOLD` | Whether to add the differentiable structure constraint | Enabling significantly increases memory/time; turn off if you only care about kcat |
| `LAMBDA_STRUCT` | Structure loss weight (relative to kcat, default 0.2) | Larger = more emphasis on structural stability, may sacrifice kcat |
| `STRUCTURE_LOSS_TYPE` | Which structural metric to use | `plddt`/`ptm`/`combined` |
| `PLDDT_THRESHOLD` / `PTM_THRESHOLD` | Constraint thresholds (penalized only below, hinge) | pLDDT commonly 70, pTM 0.5 |
| `STRUCTURE_LOSS_FUNCTION` | Hardness of the penalty function | `relu`(hinge)/`softplus`/`soft_hinge` |
| `SOFTPLUS_SIGMA` / `SOFT_HINGE_MARGIN` | Shape parameters for the corresponding function | Keep the defaults |
| `ESMFOLD_BF16` / `ESMFOLD_NO_RECYCLES` / `ESMFOLD_CHUNK_SIZE` | Memory / speed knobs | Low memory: `BF16=True`, fewer recycles, small `CHUNK_SIZE` |
