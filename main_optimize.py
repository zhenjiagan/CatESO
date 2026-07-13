import torch
import torch.nn.functional as F
from models import CatESO
from optimizer.esm_manager import ESM2Manager
from optimizer.sequence_optimizer import SequenceOptimizer
from optimizer.opt_configs import get_cfg_defaults
from optimizer.process import process_smiles_graph
import argparse
import re
import sys
from optimizer.utils import *

import os
# Set torch hub cache directory so models load from local path,
# avoiding network downloads for ESMFold and other models.
os.environ['TORCH_HOME'] = os.environ.get(
    'ESM_CKPT_DIR',
    os.path.join(os.path.expanduser('~'), 'autodl-tmp', 'esm', 'ckpt')
)

# ==============================================================
# Ensemble model configuration mapping
# ==============================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIGS_BASE_DIR = os.path.join(_PROJECT_ROOT, 'configs')

# M1 -> configs/model1 (ConcatFusion: FUSION.IS_CONCAT=True, DECODER.IN_DIM=256)
# M2 -> configs/model2 (no concat:    FUSION.IS_CONCAT=False, DECODER.IN_DIM=128)
# Same mapping inference.py uses; the shipped ensemble_ckpt weights load against
# these two architectures with no size mismatch.
ENSEMBLE_MODEL_CONFIGS = {
    'M1': 'model1',
    'M2': 'model2'
}


def _build_config_dict(cfg):
    """Build the full config dict for CatESO from a yacs CfgNode."""
    return {
        'DRUG':    dict(cfg.DRUG),
        'PROTEIN': dict(cfg.PROTEIN),
        'DECODER': dict(cfg.DECODER),
        'ICFE':    dict(cfg.ICFE),
        'MHSA':    dict(cfg.MHSA),
        'MCDC':    dict(cfg.MCDC),
        'FUSION':  dict(cfg.FUSION),
    }


def parse_ensemble_key(checkpoint_path):
    """Parse the ensemble key (M1~M5) from a checkpoint filename.

    Supported naming format: model_{seed_idx}_M{k}.pth
    Examples: model_1_M1.pth → 'M1'
              model_2_M3.pth → 'M3'

    Returns:
        str | None: 'M1'~'M5', or None if the filename does not match the convention.
    """
    filename = os.path.basename(checkpoint_path)
    match = re.search(r'model_\d+_(M[1-5])\.pth$', filename)
    return match.group(1) if match else None


def discover_ensemble_checkpoints(ckpt_dir):
    """Scan a directory and return checkpoint paths sorted by M1→M5 then seed ascending.

    Only collects files whose names match the model_{seed_idx}_M{k}.pth convention.

    Args:
        ckpt_dir: Path to the ensemble_ckpt directory.

    Returns:
        list[str]: Sorted list of absolute checkpoint paths.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"ensemble_ckpt directory not found: {ckpt_dir}")

    pattern = re.compile(r'model_(\d+)_(M[1-5])\.pth$')
    found = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            seed_idx = int(m.group(1))
            model_key = m.group(2)
            found.append((model_key, seed_idx, os.path.join(ckpt_dir, fname)))

    if not found:
        raise FileNotFoundError(
            f"No files matching model_{{N}}_M{{k}}.pth found in {ckpt_dir}"
        )

    # Sort by M1→M5 then seed ascending to ensure deterministic ordering.
    found.sort(key=lambda x: (x[0], x[1]))
    paths = [item[2] for item in found]

    print(f"\n[Auto-discovered Ensemble Checkpoints] Directory: {ckpt_dir}")
    print(f"{'Filename':<40} {'Model key':<6} {'Hyperparameter config'}")
    print("-" * 80)
    for model_key, seed_idx, path in found:
        config_name = ENSEMBLE_MODEL_CONFIGS[model_key]
        print(f"  {os.path.basename(path):<38} {model_key:<6} {config_name}")
    print(f"Found {len(paths)} checkpoint(s)\n")

    return paths


def load_model_config(checkpoint_path, configs_base_dir=DEFAULT_CONFIGS_BASE_DIR):
    """Load the hyperparameter config corresponding to a checkpoint filename.

    Naming convention: model_{seed_idx}_M{k}.pth
    - model_1_M1.pth and model_2_M1.pth share the same hyperparameters (M1 → Bayesian.yaml)
    - 5 hyperparameter sets (M1~M5), each with 2 seed variants, giving 10 checkpoints total.

    Returns:
        dict: Full config dict that can be passed directly to CatESO(**config).
    """
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from run.configs import get_cfg_defaults as _get_model_cfg_defaults

    cfg = _get_model_cfg_defaults()

    model_key = parse_ensemble_key(checkpoint_path)
    if model_key is not None:
        config_name = ENSEMBLE_MODEL_CONFIGS[model_key]
        model_yaml = os.path.join(configs_base_dir, config_name, 'model.yaml')
        data_yaml = os.path.join(configs_base_dir, config_name, 'data.yaml')
        if not os.path.exists(model_yaml):
            raise FileNotFoundError(
                f"Hyperparameter file not found: {model_yaml}\n"
                f"Ensure --ensemble_configs_dir points to the correct directory "
                f"and that {config_name}/model.yaml exists inside it."
            )
        cfg.merge_from_file(model_yaml)
        if os.path.exists(data_yaml):
            cfg.merge_from_file(data_yaml)
        print(f"  [config] {os.path.basename(checkpoint_path)} → {model_key} → {config_name}/model.yaml"
              + (f" + {config_name}/data.yaml" if os.path.exists(data_yaml) else ""))
    else:
        print(f"  [config] Non-standard ensemble filename, using default config from run/configs.py: {os.path.basename(checkpoint_path)}")

    return _build_config_dict(cfg)

def optimize_sequence(checkpoint_paths, output, num_iterations, lr, log_interval, init_sequence,
                      fixed_indices, target_smiles, cfg, seed, optimizer_type='adam',
                      optimizer_kwargs=None, penalty_type=None, penalty_lambda=0.0,
                      ste_interval=None, ste_mode='none',
                      tau_init=1.0, tau_min=0.5,
                      anneal=False, annealing_scheme='linear',
                      print_opt=False, early_stop=False, patience=5,
                      # ESMFold structure-constraint parameters
                      use_ste=False, use_esmfold=False, lambda_struct=0.5,
                      plddt_threshold=70.0, ptm_threshold=0.5, structure_loss_type='plddt',
                      structure_loss_function='relu', softplus_sigma=1.0, esmfold_bf16=True, esmfold_no_recycles=0,
                      esmfold_chunk_size=64,
                      # learning-rate decay parameters
                      lr_scheduler='none', lr_min=1e-6,
                      # STE enhancement parameters
                      decoupled_ste=False,
                      # kcat loss parameters
                      kcat_loss_type='mean_only',
                      kcat_uncertainty_beta=0.1,
                      # ensemble hyperparameter-config directory
                      ensemble_configs_dir=DEFAULT_CONFIGS_BASE_DIR):
    """
    Run sequence optimization.

    Args:
        checkpoint_paths: List of model checkpoint paths (supports single or ensemble).
        output: Output directory.
        num_iterations: Number of optimization iterations.
        lr: Learning rate.
        log_interval: Console logging interval.
        init_sequence: Initial amino-acid sequence.
        fixed_indices: List of position indices to fix.
        target_smiles: Target molecule SMILES string.
        cfg: Configuration object.
        seed: Random seed.
        optimizer_type: Optimizer type.
        optimizer_kwargs: Extra keyword arguments for the optimizer.
        penalty_type: Penalty term type ('kl' or 'l2_embedding').
        penalty_lambda: Penalty coefficient.
        ste_interval: Straight-Through Estimator (STE) application interval.
        ste_mode: STE mode ('none', 'hard', 'reinit').
        tau_init: Initial temperature.
        tau_min: Minimum temperature.
        anneal: Whether to use temperature annealing.
        annealing_scheme: Annealing scheme ('linear', 'cosine', 'exp', 'fixed').
        print_opt: Whether to write detailed per-iteration logs to CSV.
        early_stop: Whether to enable early stopping.
        patience: Early stopping patience.
        use_ste: Whether to use STE for predictions.
        use_esmfold: Whether to enable ESMFold structural constraints.
        lambda_struct: Structural loss weight.
        plddt_threshold: pLDDT threshold (0-100).
        ptm_threshold: pTM threshold (0-1).
        structure_loss_type: Structural loss type ('plddt', 'ptm', 'combined').
        structure_loss_function: Structural loss function ('relu', 'softplus').
        softplus_sigma: Sigma parameter for Softplus (only when structure_loss_function='softplus').
        esmfold_bf16: Whether to use BF16 precision for ESMFold inference.
        lr_scheduler: LR decay scheme ('none', 'linear', 'cosine').
        lr_min: Minimum learning rate (used when lr_scheduler != 'none').
    """
    # 0. Set random seed for reproducibility.
    set_seed(seed, full_deterministic=True)

    # 1. Validate that the sequence contains only standard amino acids.
    if not check_sequence_is_standard_aa(init_sequence):
        raise ValueError("Sequence contains non-standard amino acids. Please use standard amino acids only.")

    # 2. Select device (GPU preferred).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 4. Create and load model(s) (supports single model or ensemble).
    model_list = []
    print(f"\n{'='*60}")
    print(f"Loading models ({len(checkpoint_paths)} total)")
    print(f"{'='*60}")

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n[Model {i+1}/{len(checkpoint_paths)}] Loading: {checkpoint_path}")

        # Each checkpoint uses its own independent hyperparameter config.
        model_config = load_model_config(checkpoint_path, configs_base_dir=ensemble_configs_dir)

        cateso_model = CatESO(**model_config)

        # Checkpoint files store the state_dict directly.
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        cateso_model.load_state_dict(state_dict)
        cateso_model = cateso_model.to(device)
        cateso_model.eval()
        model_list.append(cateso_model)
        print(f"  ✓ Model {i+1} loaded successfully!")

    if len(model_list) == 1:
        model = model_list[0]
        print(f"\n✓ Single-model mode")
    else:
        model = model_list
        print(f"\n✓ Ensemble mode ({len(model_list)} models)")

    # 5. Create the ESM-2 manager.
    esm_manager = ESM2Manager()
    esm_manager.model = esm_manager.model.to(device)
    print("✓ ESM-2 model moved to GPU!")

    # 6. Prepare the small molecule using the same featurizer as training.
    molecule_graph = process_smiles_graph(target_smiles)

    # 7. Convert index list to a fixed-positions dict.
    fixed_positions = create_fixed_positions_from_indices(init_sequence, fixed_indices)

    # 7.5 Initialize ESMFold evaluator if enabled.
    esmfold_evaluator = None
    if use_esmfold:
        print(f"\n[Initializing ESMFold evaluator]")
        try:
            from optimizer.esmfold_evaluator import ESMFoldEvaluator

            esmfold_evaluator = ESMFoldEvaluator(
                device=device,
                chunk_size=esmfold_chunk_size,
                use_bf16=esmfold_bf16,
                no_recycles=esmfold_no_recycles
            )
            print(f"✓ ESMFold evaluator initialized successfully")

            # VRAM optimization: ESMFold already contains a full ESM2-3B internally.
            # Share it with esm_manager to avoid keeping two copies on GPU.
            print(f"\n[VRAM optimization] Sharing ESMFold's internal ESM-2 model...")
            mem_before = torch.cuda.memory_allocated(device) / 1024**3
            old_esm_model = esm_manager.model
            esm_manager.model = esmfold_evaluator.model.esm
            del old_esm_model
            torch.cuda.empty_cache()
            mem_after = torch.cuda.memory_allocated(device) / 1024**3
            print(f"  ✓ Shared. Released {mem_before - mem_after:.2f} GB VRAM")

        except Exception as e:
            print(f"[Warning] ESMFold initialization failed: {e}")
            print(f"[Warning] Structural constraints will be disabled")
            use_esmfold = False

    # 8. Create the sequence optimizer.
    print(f"\nCreating sequence optimizer...")
    print(f"  - Target molecule: {target_smiles}")
    print(f"  - Initial sequence length: {len(init_sequence)}")
    print(f"  - Fixed positions: {len(fixed_positions)}")
    print(f"  - Optimizable positions: {len(init_sequence) - len(fixed_positions)}")
    print(f"  - Optimizer type: {optimizer_type}")
    print(f"  - Penalty type: {penalty_type if penalty_type else 'none'}")
    print(f"  - Penalty coefficient: {penalty_lambda}")
    print(f"  - Use STE (Straight-Through Estimator): {'yes' if use_ste else 'no'}")
    print(f"  - ESMFold structural constraint: {'enabled' if use_esmfold else 'disabled'}")
    if use_esmfold:
        print(f"    * Structural loss weight: {lambda_struct}")
        print(f"    * pLDDT threshold: {plddt_threshold}")
        print(f"    * pTM threshold: {ptm_threshold}")
        print(f"    * Structural loss type: {structure_loss_type}")
        print(f"    * Structural loss function: {structure_loss_function}")
        if structure_loss_function == 'softplus':
            print(f"    * Softplus σ: {softplus_sigma}")
    if ste_interval and ste_mode != 'none':
        print(f"  - STE interval: {ste_interval}")
        print(f"  - STE mode: {ste_mode}")
    print(f"  - Decoupled STE: {'yes' if decoupled_ste else 'no'}")
    print(f"  - kcat loss type: {kcat_loss_type}")
    if kcat_loss_type == 'uncertainty_aware':
        print(f"    * Uncertainty penalty coefficient β: {kcat_uncertainty_beta}")

    seq_optimizer = SequenceOptimizer(
        model=model,
        esm_manager=esm_manager,
        seq_length=len(init_sequence),
        molecule_graph=molecule_graph,
        init_sequence=init_sequence,
        fixed_positions=fixed_positions,
        device=device,
        penalty_type=penalty_type,
        penalty_lambda=penalty_lambda,
        use_esmfold=use_esmfold,
        esmfold_evaluator=esmfold_evaluator,
        lambda_struct=lambda_struct,
        plddt_threshold=plddt_threshold,
        ptm_threshold=ptm_threshold,
        structure_loss_type=structure_loss_type,
        structure_loss_function=structure_loss_function,
        softplus_sigma=softplus_sigma,
        use_ste=use_ste,
        decoupled_ste=decoupled_ste,
        kcat_loss_type=kcat_loss_type,
        kcat_uncertainty_beta=kcat_uncertainty_beta
    )
    print("✓ Sequence optimizer created successfully!")

    # 9. Run optimization.
    if anneal:
        print(f"\nStarting sequence optimization ({num_iterations} iterations, annealing: {annealing_scheme}, tau={tau_init}->{tau_min})...")
    else:
        print(f"\nStarting sequence optimization ({num_iterations} iterations, fixed tau={tau_init})...")
    print("=" * 60)
    optimized_sequence, history, logits_matrix, probs_matrix, aa_mapping = seq_optimizer.optimize(
        num_iterations=num_iterations,
        lr=lr,
        log_interval=log_interval,
        tau_init=tau_init,
        tau_min=tau_min,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        ste_interval=ste_interval,
        ste_mode=ste_mode,
        anneal=anneal,
        annealing_scheme=annealing_scheme,
        print_opt=print_opt,
        early_stop=early_stop,
        patience=patience,
        output_dir=output, # Pass output dir for CSV logging
        seed=seed,
        lr_scheduler=lr_scheduler,
        lr_min=lr_min
    )
    print("=" * 60)

    # 10. Decode the optimized sequence.
    alphabet = esm_manager.alphabet
    optimized_seq_str = ''.join([
        alphabet.get_tok(idx.item())
        for idx in optimized_sequence.cpu()
    ])

    print(f"\n✓ Optimization complete!")
    print(f"Initial sequence:   {init_sequence}")
    print(f"Optimized sequence: {optimized_seq_str}")

    # ===== Mutation analysis =====
    print(f"\n{'='*80}")
    print(f"Mutation Analysis")
    print(f"{'='*80}")

    mutations = []
    for i, (init_aa, opt_aa) in enumerate(zip(init_sequence, optimized_seq_str)):
        if init_aa != opt_aa:
            mutations.append({
                'position': i,
                'from': init_aa,
                'to': opt_aa,
                'is_fixed': i in fixed_positions
            })

    num_mutations = len(mutations)
    num_mutable_positions = len(init_sequence) - len(fixed_positions)
    mutation_rate = (num_mutations / num_mutable_positions * 100) if num_mutable_positions > 0 else 0

    print(f"\nSequence length: {len(init_sequence)}")
    print(f"Fixed positions: {len(fixed_positions)}")
    print(f"Mutable positions: {num_mutable_positions}")
    print(f"\nMutated positions: {num_mutations}")
    print(f"Mutation rate: {mutation_rate:.2f}% ({num_mutations}/{num_mutable_positions})")

    if num_mutations > 0:
        print(f"\nMutation details ({num_mutations} position(s)):")
        print(f"{'Position':<8} {'From':<6} {'To':<8} {'Mutation':<15}")
        print(f"{'-'*8} {'-'*6} {'-'*8} {'-'*15}")

        for mut in mutations:
            mutation_str = f"{mut['from']}{mut['position']}{mut['to']}"
            if mut['is_fixed']:
                # Safety check — this should never happen
                print(f"{mut['position']:<8} {mut['from']:<6} {mut['to']:<8} {mutation_str:<15} [WARNING: fixed position changed!]")
            else:
                print(f"{mut['position']:<8} {mut['from']:<6} {mut['to']:<8} {mutation_str:<15}")

        if num_mutations <= 20:
            mutation_summary = ', '.join([f"{m['from']}{m['position']}{m['to']}" for m in mutations])
        else:
            mutation_summary = ', '.join([f"{m['from']}{m['position']}{m['to']}" for m in mutations[:10]]) + f" ... ({num_mutations} total)"
        print(f"\nMutation summary: {mutation_summary}")
    else:
        print(f"\nNo mutations (optimized sequence is identical to the initial sequence)")

    print(f"{'='*80}\n")

    # 11. Verify fixed positions are unchanged.
    check_fixed_positions(optimized_sequence, fixed_positions, alphabet)

    # 12. Compute final prediction.
    with torch.no_grad():
        final_result = seq_optimizer.soft_embedding_forward(
            F.one_hot(optimized_sequence, num_classes=len(alphabet)).float().to(device)
        )
        if isinstance(final_result, tuple):
            final_kcat, final_std = final_result
        else:
            final_kcat = final_result
            final_std = None
            
    print(f"\nFinal predicted kcat: {final_kcat.item():.4f}")
    if final_std is not None:
        print(f"  - Prediction std: {final_std.item():.4f}")
    if len(model_list) > 1:
        print(f"  (Ensemble average over {len(model_list)} models)")

    # 13. Save results (including full logits and probability matrices).
    output_file = os.path.join(output, f'optimize_output_{seed}.pth')
    torch.save({
        'optimized_sequence': optimized_seq_str,
        'kcat_pred': final_kcat.item(),
        'kcat_std': final_std.item() if final_std is not None else 0.0,
        'history': history,
        'target_smiles': target_smiles,
        'fixed_positions': fixed_positions,
        'init_sequence': init_sequence,
        'logits_matrix': logits_matrix.cpu(),  # [L, 20]
        'probs_matrix': probs_matrix.cpu(),    # [L, 20]
        'aa_mapping': aa_mapping,              # ordering of the 20 standard amino acids
        'device': str(device),
        'seed': seed,
        'num_iterations': num_iterations,
        'lr': lr,
        'optimizer_type': optimizer_type,
        'optimizer_kwargs': optimizer_kwargs,
        'penalty_type': penalty_type,
        'penalty_lambda': penalty_lambda,
        'num_models': len(model_list),
        'checkpoint_paths': checkpoint_paths,
        'ste_interval': ste_interval,
        'ste_mode': ste_mode,
        'mutations': mutations,
        'num_mutations': num_mutations,
        'mutation_rate': mutation_rate,
        'anneal': anneal,
        'annealing_scheme': annealing_scheme,
        'early_stop': early_stop,
        'patience': patience,
        'kcat_loss_type': kcat_loss_type,
        'kcat_uncertainty_beta': kcat_uncertainty_beta
    }, output_file)
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  - Logits matrix shape: {logits_matrix.shape}")
    print(f"  - Probability matrix shape: {probs_matrix.shape}")
    print(f"  - Amino acid mapping: {aa_mapping}")
    print(f"  - Random seed: {seed}")
    print(f"  - Optimizer type: {optimizer_type}")
    print(f"  - Penalty type: {penalty_type if penalty_type else 'none'}")
    print(f"  - Penalty coefficient: {penalty_lambda}")
    print(f"  - Ensemble model count: {len(model_list)}")

    # 14. Free VRAM cache between seeds.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"\n✓ GPU cache cleared")

def parse_args():
    parser = argparse.ArgumentParser(description='Sequence Optimization with Ensemble Models and Penalty Terms')
    
    parser.add_argument('--opt_yaml', type=str, required=True,
                        help='Path to optimization config YAML file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model config YAML file')
    parser.add_argument(
        '--ensemble_configs_dir',
        type=str,
        default=DEFAULT_CONFIGS_BASE_DIR,
        help=(
            'Directory containing ensemble hyperparameter sub-folders, '
            'each with a Bayesian.yaml. Default: <script_dir>/configs'
        )
    )
    parser.add_argument(
        '--ensemble_ckpt_dir',
        type=str,
        default=None,
        help=(
            'Ensemble checkpoint directory containing all model_{N}_M{k}.pth files. '
            'When specified, the directory is scanned automatically and overrides '
            'OPTIMIZATION.CHECKPOINT in the YAML. Default: None (use YAML CHECKPOINT).'
        )
    )

    return parser.parse_args()

def main():
    args = parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.model)
    cfg.merge_from_file(args.opt_yaml)

    print(f"\n{'='*80}")
    print(f"Sequence Optimization Configuration Summary")
    print(f"{'='*80}")
    print(f"\n[Config files]")
    print(f"  - Model config: {args.model}")
    print(f"  - Optimization config: {args.opt_yaml}")
    print(f"  - Ensemble hyperparameter dir: {args.ensemble_configs_dir}")
    if args.ensemble_ckpt_dir:
        print(f"  - Ensemble checkpoint dir: {args.ensemble_ckpt_dir} (auto-scan mode)")

    # If --ensemble_ckpt_dir is provided, auto-discover checkpoints (ignores YAML CHECKPOINT).
    if args.ensemble_ckpt_dir:
        checkpoint_paths = discover_ensemble_checkpoints(args.ensemble_ckpt_dir)
    else:
        checkpoint_paths = parse_checkpoint_paths(cfg.OPTIMIZATION.CHECKPOINT)
    output = cfg.OPTIMIZATION.OUTPUT
    num_iterations = cfg.OPTIMIZATION.NUM_ITERATIONS
    lr = cfg.OPTIMIZATION.LR
    log_interval = cfg.OPTIMIZATION.LOG_INTERVAL
    init_sequence = cfg.OPTIMIZATION.INIT_SEQUENCE
    target_smiles = cfg.OPTIMIZATION.TARGET_SMILES

    optimizer_type = cfg.OPTIMIZATION.OPTIMIZER
    optimizer_kwargs = parse_optimizer_kwargs(cfg.OPTIMIZATION.OPTIMIZER_KWARGS)

    penalty_type = cfg.OPTIMIZATION.PENALTY_TYPE if cfg.OPTIMIZATION.PENALTY_TYPE else None
    penalty_lambda = cfg.OPTIMIZATION.PENALTY_LAMBDA

    ste_interval = cfg.OPTIMIZATION.STE_INTERVAL
    ste_mode = cfg.OPTIMIZATION.STE_MODE

    anneal = cfg.OPTIMIZATION.ANNEAL
    annealing_scheme = cfg.OPTIMIZATION.ANNEALING_SCHEME
    tau_init = cfg.OPTIMIZATION.TEMPERATURE
    tau_min = cfg.OPTIMIZATION.MIN_TEM

    print_opt = cfg.OPTIMIZATION.PRINT
    early_stop = cfg.OPTIMIZATION.EARLY_STOP
    patience = cfg.OPTIMIZATION.PATIENCE

    use_ste = cfg.OPTIMIZATION.USE_STE
    use_esmfold = cfg.OPTIMIZATION.USE_ESMFOLD
    lambda_struct = cfg.OPTIMIZATION.LAMBDA_STRUCT
    plddt_threshold = cfg.OPTIMIZATION.PLDDT_THRESHOLD
    ptm_threshold = cfg.OPTIMIZATION.PTM_THRESHOLD
    structure_loss_type = cfg.OPTIMIZATION.STRUCTURE_LOSS_TYPE
    structure_loss_function = cfg.OPTIMIZATION.STRUCTURE_LOSS_FUNCTION
    softplus_sigma = cfg.OPTIMIZATION.SOFTPLUS_SIGMA
    esmfold_bf16 = cfg.OPTIMIZATION.ESMFOLD_BF16
    esmfold_no_recycles = cfg.OPTIMIZATION.ESMFOLD_NO_RECYCLES
    esmfold_chunk_size = cfg.OPTIMIZATION.ESMFOLD_CHUNK_SIZE if cfg.OPTIMIZATION.ESMFOLD_CHUNK_SIZE > 0 else None

    lr_scheduler = cfg.OPTIMIZATION.LR_SCHEDULER
    lr_min = cfg.OPTIMIZATION.LR_MIN

    decoupled_ste = cfg.OPTIMIZATION.DECOUPLED_STE

    kcat_loss_type = cfg.OPTIMIZATION.KCAT_LOSS_TYPE
    kcat_uncertainty_beta = cfg.OPTIMIZATION.KCAT_UNCERTAINTY_BETA

    seeds = cfg.OPTIMIZATION.SEED if isinstance(cfg.OPTIMIZATION.SEED, list) else [cfg.OPTIMIZATION.SEED]

    fixed_indices = parse_fixed_indices(cfg.OPTIMIZATION.FIXED_INDICES)

    print(f"\n[Model config]")
    print(f"  - Checkpoint path(s): {checkpoint_paths}")
    print(f"  - Model count: {len(checkpoint_paths)} {'(ensemble mode)' if len(checkpoint_paths) > 1 else '(single model)'}")

    print(f"\n[Optimizer config]")
    print(f"  - Optimizer type: {optimizer_type}")
    print(f"  - Initial LR: {lr}")
    if lr_scheduler != 'none':
        print(f"  - LR scheduler: {lr_scheduler} ({lr} -> {lr_min})")
    else:
        print(f"  - LR: fixed")
    if optimizer_kwargs:
        print(f"  - Extra kwargs: {optimizer_kwargs}")

    print(f"\n[Penalty config]")
    print(f"  - Penalty type: {penalty_type if penalty_type else 'none'}")
    print(f"  - Penalty coefficient (lambda): {penalty_lambda}")

    print(f"\n[STE config]")
    print(f"  - STE interval: {ste_interval if ste_interval else 'disabled'}")
    print(f"  - STE mode: {ste_mode}")
    print(f"  - Decoupled STE: {'yes' if decoupled_ste else 'no'}")
    print(f"  - kcat loss type: {kcat_loss_type}")
    if kcat_loss_type == 'uncertainty_aware':
        print(f"    * Uncertainty penalty coefficient β: {kcat_uncertainty_beta}")

    print(f"\n[Temperature annealing config]")
    print(f"  - Annealing enabled: {'yes' if anneal else 'no'}")
    if anneal:
        print(f"  - Annealing scheme: {annealing_scheme}")
        print(f"  - Initial temperature: {tau_init}")
        if annealing_scheme != 'fixed':
            print(f"  - Minimum temperature: {tau_min}")
    else:
        print(f"  - Fixed temperature: {tau_init}")

    print(f"\n[ESMFold structural constraint config]")
    print(f"  - Use STE (Straight-Through Estimator): {'yes' if use_ste else 'no'}")
    print(f"  - ESMFold structural constraint: {'enabled' if use_esmfold else 'disabled'}")
    if use_esmfold:
        print(f"  - Structural loss weight: {lambda_struct}")
        print(f"  - pLDDT threshold: {plddt_threshold}")
        print(f"  - pTM threshold: {ptm_threshold}")
        print(f"  - Structural loss type: {structure_loss_type}")
        print(f"  - Structural loss function: {structure_loss_function}")
        if structure_loss_function == 'softplus':
            print(f"  - Softplus σ: {softplus_sigma}")
        print(f"  - Use BF16 precision: {'yes' if esmfold_bf16 else 'no'}")
        print(f"  - Trunk recycles: {esmfold_no_recycles} ({esmfold_no_recycles + 1} pass(es) total)")
        print(f"  - Chunk size: {esmfold_chunk_size if esmfold_chunk_size is not None else 'disabled (full computation)'}")

    print(f"\n[Random seeds]")
    print(f"  - Config value: {cfg.OPTIMIZATION.SEED}")
    print(f"  - Seed list: {seeds}")
    print(f"  - Total seeds: {len(seeds)}")

    print(f"\n[Fixed positions]")
    print(f"  - Input string: '{cfg.OPTIMIZATION.FIXED_INDICES}'")
    print(f"  - Parsed: {fixed_indices[:10]}{'...' if len(fixed_indices) > 10 else ''} ({len(fixed_indices)} position(s))")

    print(f"\n[Optimization parameters]")
    print(f"  - Iterations: {num_iterations}")
    print(f"  - Log interval: {log_interval}")
    print(f"  - Initial sequence length: {len(init_sequence)}")
    print(f"  - Target molecule: {target_smiles}")
    print(f"{'='*80}\n")

    if not os.path.exists(output):
        os.makedirs(output)
        print(f"\n✓ Created output directory: {output}")

    print(f"\n{'='*80}")
    print(f"Starting batch optimization ({len(seeds)} seed(s))")
    print(f"{'='*80}\n")

    for idx, seed in enumerate(seeds, 1):
        print(f"\n{'#'*80}")
        print(f"# Run {idx}/{len(seeds)}: seed = {seed}")
        print(f"{'#'*80}\n")

        try:
            optimize_sequence(
                checkpoint_paths=checkpoint_paths,
                output=output,
                num_iterations=num_iterations,
                lr=lr,
                log_interval=log_interval,
                init_sequence=init_sequence,
                fixed_indices=fixed_indices,
                target_smiles=target_smiles,
                cfg=cfg,
                seed=seed,
                optimizer_type=optimizer_type,
                optimizer_kwargs=optimizer_kwargs,
                penalty_type=penalty_type,
                penalty_lambda=penalty_lambda,
                ste_interval=ste_interval,
                ste_mode=ste_mode,
                tau_init=tau_init,
                tau_min=tau_min,
                anneal=anneal,
                annealing_scheme=annealing_scheme,
                print_opt=print_opt,
                early_stop=early_stop,
                patience=patience,
                use_ste=use_ste,
                use_esmfold=use_esmfold,
                lambda_struct=lambda_struct,
                plddt_threshold=plddt_threshold,
                ptm_threshold=ptm_threshold,
                structure_loss_type=structure_loss_type,
                structure_loss_function=structure_loss_function,
                softplus_sigma=softplus_sigma,
                esmfold_bf16=esmfold_bf16,
                esmfold_no_recycles=esmfold_no_recycles,
                esmfold_chunk_size=esmfold_chunk_size,
                lr_scheduler=lr_scheduler,
                lr_min=lr_min,
                decoupled_ste=decoupled_ste,
                kcat_loss_type=kcat_loss_type,
                kcat_uncertainty_beta=kcat_uncertainty_beta,
                ensemble_configs_dir=args.ensemble_configs_dir
            )
            print(f"\n✓ Seed {seed} optimization complete!")
        except Exception as e:
            print(f"\n✗ Seed {seed} optimization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"All optimization runs complete!")
    print(f"  - Seeds run: {len(seeds)}")
    print(f"  - Output directory: {output}")
    print(f"  - Output filename pattern: optimize_output_{{seed}}.pth")
    print(f"  - Optimizer type: {optimizer_type}")
    print(f"  - Penalty type: {penalty_type if penalty_type else 'none'}")
    print(f"  - Penalty coefficient: {penalty_lambda}")
    print(f"  - Ensemble model count: {len(checkpoint_paths)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()