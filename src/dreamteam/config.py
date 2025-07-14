import os

# Gemini API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Training parameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
DEVICE = "mps"
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

# Evolutionary algorithm parameters
POPULATION_SIZE = 25
N_GENERATIONS = 10

# Paths
DATA_DIR = "data/shakespeare"
MODELS_DIR = "models"
PROMPTS_DIR = "src/dreamteam/prompts"
AGENTS_DIR = "src/dreamteam/agents"
