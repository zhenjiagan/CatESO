from yacs.config import CfgNode as CN

_C = CN()

#Substrate feature extractor
_C.DRUG = CN()
_C.DRUG.NODE_IN_FEATS = 75
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 128, 128]
_C.DRUG.NODE_IN_EMBEDDING = 128


# Enzyme feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.NAME = "ESM_MLP_3B"
_C.PROTEIN.IN_DIM = 2560
_C.PROTEIN.HIDDEN_DIM = 640
_C.PROTEIN.TARGET_DIM = 128

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 128
_C.DECODER.HIDDEN_DIM = 256
_C.DECODER.OUT_DIM = 64
_C.DECODER.BINARY = 1
_C.DECODER.DEEP = False

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.NUM_WORKERS = 4
_C.SOLVER.LR = 5e-5
_C.SOLVER.SEED = [42]
_C.SOLVER.DATA = ""
_C.SOLVER.SAVE = ""

_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = './results'

_C.CCFM = CN()
_C.CCFM.DIM = 128
_C.CCFM.RATIO = 0.5

_C.BCFM = CN()
_C.BCFM.DIM = 128

_C.STAGE = CN()
_C.STAGE.CCFM = True
_C.STAGE.BCFM = True
_C.STAGE.NUM = 1

_C.ICFE = CN()
_C.ICFE.DIM = 256
_C.ICFE.NUM_HEAD = 8
_C.ICFE.BLOCK_EXP = 4
_C.ICFE.LOOPS_NUM = 2
_C.ICFE.N_LAYER = 1
_C.ICFE.EMBD_PDROP = 0.1
_C.ICFE.ATTN_PDROP = 0.0
_C.ICFE.RESID_PDROP = 0.1
_C.ICFE.IS_ICFE = True

_C.MHSA = CN()
_C.MHSA.DIM = 128
_C.MHSA.N_LAYER = 1
_C.MHSA.N_HEAD = 4
_C.MHSA.DROP_RATE = 0.1
_C.MHSA.IS_MHSA = True

_C.MCDC = CN()
_C.MCDC.KERNEL_LIST = [3, 5, 7]
_C.MCDC.K = 4
_C.MCDC.IS_MCDC = True

_C.FUSION = CN()
_C.FUSION.IS_CONCAT = False
_C.FUSION.IS_DCPA = False
_C.FUSION.USE_ATTN_POOL = False
_C.FUSION.N_HEAD = 4

# Optimization parameters
_C.OPTIMIZATION = CN()
_C.OPTIMIZATION.CHECKPOINT = ""  # Checkpoint path(s) or directory
_C.OPTIMIZATION.OUTPUT = "./optimize_output"  # Output directory
_C.OPTIMIZATION.NUM_ITERATIONS = 1000  # Number of optimization iterations
_C.OPTIMIZATION.LR = 0.001  # Learning rate
_C.OPTIMIZATION.LOG_INTERVAL = 50  # Log interval
_C.OPTIMIZATION.INIT_SEQUENCE = ""  # Initial protein sequence
_C.OPTIMIZATION.TARGET_SMILES = ""  # Target molecule SMILES
_C.OPTIMIZATION.FIXED_INDICES = ""  # Fixed position indices (e.g., "0-49" or "0,10,20")
_C.OPTIMIZATION.SEED = [42]  # Random seed(s) as list
_C.OPTIMIZATION.OPTIMIZER = "adamw"  # Optimizer type: adam, adamw, sgd, rmsprop, adagrad
_C.OPTIMIZATION.OPTIMIZER_KWARGS = ""  # Additional optimizer kwargs (e.g., "weight_decay=0.01")
_C.OPTIMIZATION.PENALTY_TYPE = ""  # Penalty type: kl, l2_embedding, or empty string for None
_C.OPTIMIZATION.PENALTY_LAMBDA = 0.0  # Penalty coefficient
_C.OPTIMIZATION.STE_INTERVAL = None  # STE interval
_C.OPTIMIZATION.STE_MODE = "none"  # STE mode: none, hard
_C.OPTIMIZATION.KCAT_LOSS_TYPE = "mean_only"  # kcat loss type: mean_only, uncertainty_aware
_C.OPTIMIZATION.KCAT_UNCERTAINTY_BETA = 0.1  # Risk penalty coefficient beta

# Temperature Annealing Configuration
_C.OPTIMIZATION.ANNEAL = True  # Enable/disable temperature annealing
_C.OPTIMIZATION.ANNEALING_SCHEME = "cosine"  # Annealing scheme: linear, cosine, exp, fixed
_C.OPTIMIZATION.TEMPERATURE = 2.0  # Initial temperature
_C.OPTIMIZATION.MIN_TEM = 0.1  # Minimum temperature (used when ANNEAL=True and scheme != 'fixed')
_C.OPTIMIZATION.PRINT = True # For logging CSV
_C.OPTIMIZATION.EARLY_STOP = False
_C.OPTIMIZATION.PATIENCE = 5

# STE Configuration (Straight-Through Estimator)
_C.OPTIMIZATION.USE_STE = False  # Whether to use STE for kcat and structure prediction
_C.OPTIMIZATION.DECOUPLED_STE = False  # Whether to decouple sampling temperature from gradient temperature (uses tau=1.0 for gradient)
# STE: Forward pass uses hard (argmax) sequence, backward pass uses soft gradients

# ESMFold Structure Constraint Configuration (Reparameterization)
_C.OPTIMIZATION.USE_ESMFOLD = False  # Enable ESMFold structure constraint (differentiable)
_C.OPTIMIZATION.LAMBDA_STRUCT = 0.5  # Weight coefficient for structure loss
_C.OPTIMIZATION.PLDDT_THRESHOLD = 70.0  # pLDDT threshold for Hinge Loss
_C.OPTIMIZATION.PTM_THRESHOLD = 0.5  # pTM threshold for Hinge Loss (0-1 range)
_C.OPTIMIZATION.STRUCTURE_LOSS_TYPE = "plddt"  # Structure loss type: plddt, ptm, combined
_C.OPTIMIZATION.STRUCTURE_LOSS_FUNCTION = "relu"  # Structure loss function: relu (Hinge Loss), softplus, soft_hinge
_C.OPTIMIZATION.SOFTPLUS_SIGMA = 1.0  # Softplus sigma parameter (used when STRUCTURE_LOSS_FUNCTION in {softplus, soft_hinge})
_C.OPTIMIZATION.SOFT_HINGE_MARGIN = 5.0  # Soft-hinge cutoff margin in score units; loss is exactly 0 (with zero gradient) once score >= threshold + margin (only used when STRUCTURE_LOSS_FUNCTION=soft_hinge)
_C.OPTIMIZATION.ESMFOLD_BF16 = False  # Use BF16 precision for ESMFold (saves memory)
_C.OPTIMIZATION.ESMFOLD_NO_RECYCLES = 3  # ESMFold trunk recycles (0=1 pass, 3=4 passes; lower saves memory)
_C.OPTIMIZATION.ESMFOLD_CHUNK_SIZE = 0  # Triangular attention chunk size (smaller = less memory, slower; None-like: set to 0 to disable)

# Learning Rate Scheduler Configuration
_C.OPTIMIZATION.LR_SCHEDULER = "none"  # Learning rate scheduler: none, linear, cosine
_C.OPTIMIZATION.LR_MIN = 1e-6  # Minimum learning rate (used when LR_SCHEDULER != 'none')

def get_cfg_defaults():
    return _C.clone()
