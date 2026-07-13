import random
import numpy as np
import torch
import os
import glob


def parse_seeds(seeds_str):
    """
    Parse a random-seed specification string.

    Supported formats:
    - "42" -> [42]
    - "1,2,3" -> [1, 2, 3]
    - "1-5" -> [1, 2, 3, 4, 5]
    - "1,5-10,20" -> [1, 5, 6, 7, 8, 9, 10, 20]

    Args:
        seeds_str: Seed string.

    Returns:
        List of integer seeds (deduplicated, sorted).
    """
    if not seeds_str:
        return [42]  # default seed

    seeds = []
    parts = seeds_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start = int(start.strip())
            end = int(end.strip())
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))

    seeds = sorted(list(set(seeds)))
    return seeds


def parse_fixed_indices(indices_input):
    """
    Parse fixed-position indices from a string or list (single indices or range strings).

    Supported formats:
    - String: "1,2,3", "1-5", "1-5,10,15-20"
    - List: [1, 2, 3], ["1-5", 10, "15-20"]

    Args:
        indices_input: String or list/tuple of index specs.

    Returns:
        Sorted, deduplicated list of integer indices.
    """
    if not indices_input:
        return []

    if isinstance(indices_input, str):
        parts = indices_input.split(',')
    elif isinstance(indices_input, (list, tuple)):
        parts = [str(p) for p in indices_input]
    else:
        parts = [str(indices_input)]

    indices = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                if start <= end:
                    indices.extend(range(start, end + 1))
                else:
                    indices.extend(range(end, start + 1))
            except (ValueError, IndexError):
                print(f"Warning: could not parse index range '{part}', skipped")
        else:
            try:
                indices.append(int(part))
            except ValueError:
                print(f"Warning: could not parse index token '{part}', skipped")

    indices = sorted(list(set(indices)))
    return indices


def parse_checkpoint_paths(checkpoint_str):
    """
    Parse checkpoint path string or directory path.

    Supported formats:
    - Single file: "model.pth"
    - Multiple files (comma-separated): "a.pth,b.pth"
    - Directory: "/path/to/checkpoints/" -> all *.pth files in that directory

    Args:
        checkpoint_str: Path string or directory.

    Returns:
        List of checkpoint paths (sorted by filename for directory scan).
    """
    if not checkpoint_str:
        return []

    checkpoint_str = checkpoint_str.strip()

    if os.path.isdir(checkpoint_str):
        pth_pattern = os.path.join(checkpoint_str, "*.pth")
        pth_files = glob.glob(pth_pattern)

        if not pth_files:
            print(f"Warning: no .pth files found in directory '{checkpoint_str}'")
            return []

        pth_files.sort()
        print(f"Found {len(pth_files)} .pth file(s) in '{checkpoint_str}':")
        for i, path in enumerate(pth_files, 1):
            print(f"  [{i}] {os.path.basename(path)}")

        return pth_files

    paths = [p.strip() for p in checkpoint_str.split(',')]
    return [p for p in paths if p]


def parse_optimizer_kwargs(kwargs_str):
    """
    Parse optimizer extra-keyword string.

    Format: "key1=value1,key2=value2"
    Example: "momentum=0.9,weight_decay=0.01"

    Args:
        kwargs_str: Comma-separated key=value pairs.

    Returns:
        Dictionary of parsed kwargs.
    """
    if not kwargs_str:
        return {}

    kwargs = {}
    parts = kwargs_str.split(',')

    for part in parts:
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()

            try:
                if '.' in value:
                    kwargs[key] = float(value)
                else:
                    kwargs[key] = int(value)
            except ValueError:
                if value.lower() == 'true':
                    kwargs[key] = True
                elif value.lower() == 'false':
                    kwargs[key] = False
                else:
                    kwargs[key] = value

    return kwargs


def set_seed(seed: int = 42, full_deterministic: bool = False):
    """
    Set random seeds for reproducibility.

    Args:
        seed:               Random seed.
        full_deterministic: If True, enable strictest deterministic mode (may reduce speed).
    """
    print(f"{'='*60}")
    print(f"Setting random seed: {seed}  |  Full deterministic: {full_deterministic}")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if full_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            print("Strict deterministic mode enabled (may be slower).")
        else:
            print("Standard deterministic mode (recommended).")

    print("Random seed setup done.\n")


def check_fixed_positions(optimized_sequence, fixed_positions, alphabet):
    """
    Verify that fixed positions still match the intended amino acids.

    Args:
        optimized_sequence: torch.Tensor or str — optimized sequence.
        fixed_positions:    Dict {position: one-letter AA}.
        alphabet:           ESM alphabet (tensor indices -> token string).
    """
    print(f"\nChecking fixed positions:")
    all_fixed = True
    for pos, aa in fixed_positions.items():
        if isinstance(optimized_sequence, torch.Tensor):
            actual_aa = alphabet.get_tok(optimized_sequence[pos].item())
        else:
            actual_aa = optimized_sequence[pos]

        is_correct = actual_aa == aa
        all_fixed = all_fixed and is_correct
        status = "OK" if is_correct else "FAIL"
        print(f"  [{status}] position {pos}: {actual_aa} (expected {aa})")
    if all_fixed:
        print("All fixed positions match.")
    else:
        print("Some fixed positions do not match.")


def check_sequence_is_standard_aa(sequence):
    standard_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    is_standard_aa = True
    for c in sequence:
        if c not in standard_aa:
            is_standard_aa = False
            break
    return is_standard_aa


def create_fixed_positions_from_indices(init_sequence, fixed_indices):
    """
    Build a fixed_positions dict from an initial sequence string and index list.

    Args:
        init_sequence: Initial protein sequence string, e.g. "MSKGEELFTGVV..."
        fixed_indices: 0-based indices to fix, e.g. [0, 10, 25, 50, 100]

    Returns:
        Dict {pos: 'AA'}, e.g. {0: 'M', 10: 'E', ...}

    Example:
        >>> init_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK..."
        >>> fixed_idx = [0, 10, 25]
        >>> fixed_pos = create_fixed_positions_from_indices(init_seq, fixed_idx)
        >>> fixed_pos
        {0: 'M', 10: 'T', 25: 'S'}
    """
    if not isinstance(init_sequence, str):
        raise TypeError("init_sequence must be a string")

    if not isinstance(fixed_indices, (list, tuple)):
        raise TypeError("fixed_indices must be a list or tuple")

    fixed_positions = {}

    for idx in fixed_indices:
        if not isinstance(idx, int):
            raise TypeError(f"fixed_indices entries must be int, got {type(idx)}")

        if idx < 0 or idx >= len(init_sequence):
            raise ValueError(
                f"index {idx} out of range [0, {len(init_sequence) - 1}]"
            )

        amino_acid = init_sequence[idx]
        fixed_positions[idx] = amino_acid

    print(f"\n[create_fixed_positions_from_indices] Built fixed_positions dict:")
    print(f"  - init_sequence length: {len(init_sequence)}")
    print(f"  - number of fixed positions: {len(fixed_indices)}")
    print(f"  - entries:")
    for pos in sorted(fixed_positions.keys()):
        print(f"    * position {pos}: '{fixed_positions[pos]}'")

    return fixed_positions
