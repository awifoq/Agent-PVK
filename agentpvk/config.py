"""
Flow Matching Molecule Generation — Configuration
"""
from pathlib import Path

ROOT = Path(__file__).parent

# ── Data ──────────────────────────────────────────────
# 论文私有实验数据不随仓库分发；以下默认指向仓库内 data/ 目录，
# 缺失时由调用方通过命令行参数显式指定。
REPORT_100_CSV = ROOT.parent / 'data' / 'report_100_molecules.csv'
GDB9_CSV      = ROOT / 'data' / 'gdb9_additives_from_xyz_final.csv'
EXP_TXT       = ROOT.parent / 'data' / 'seed_library_46.csv'
TRAIN_CSV     = ROOT / 'data' / 'train.csv'
VAL_CSV       = ROOT / 'data' / 'val.csv'

# Data construction
GDB9_EXCELLENT_SAMPLES = 800
GDB9_GOOD_SAMPLES      = 400
RANDOM_AUGMENT_COUNT   = 800
TRAIN_SIZE_TARGET      = 2500
MAX_SMILES_LEN         = 128

# ── Model Architecture ────────────────────────────────
VOCAB_EXTRA = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
D_MODEL     = 256
N_HEAD      = 8
N_LAYER     = 4
D_FF        = 1024
DROPOUT     = 0.1
MAX_SEQ_LEN = 128

# ── Training ──────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 40
LR          = 3e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP   = 1.0
SAVE_EVERY  = 5
NUM_WORKERS = 0          # set to 0 for Windows safety

# ── Generation ────────────────────────────────────────
GEN_SAMPLES    = 500
GEN_STEPS      = 100     # ODE integration steps
GEN_TEMPERATURE = 1.0
GEN_NOISE      = 0.5     # noise_scale for from_known variant

# ── Paths ─────────────────────────────────────────────
CHECKPOINT_DIR = ROOT / 'checkpoints'
BEST_MODEL     = CHECKPOINT_DIR / 'best_model.pt'
TOKENIZER_FILE = CHECKPOINT_DIR / 'tokenizer.json'
GEN_OUTPUT     = ROOT / 'generated' / 'candidates.txt'
