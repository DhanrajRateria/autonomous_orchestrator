SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000
NUM_ROUNDS = 5
CLIENTS_PER_ROUND = 2 # Number of clients to participate in each round
MIN_CLIENTS_FOR_AGGREGATION = 2 # Minimum clients needed to proceed with aggregation
HE_KEY_SIZE = 1024 # Bits for Paillier key pair (use >=2048 in production)
MODEL_FEATURE_COUNT = 10 # Number of features in the security data
PRECISION_FACTOR = 10**6 # Factor to scale floats to integers for HE

# --- Simulated Autonomous Action ---
THREAT_THRESHOLD = 0.8 # If avg threat probability > threshold, trigger action

# --- Quantum-Inspired Enhancements ---
ENABLE_QKD_SIMULATION = True # Flag to enable/disable simulated QKD
QKD_KEY_LENGTH = 256 # Simulated key length (bits) after reconciliation

ENABLE_QIFA_MOMENTUM = True # Flag to enable/disable QIFA momentum component
QIFA_MOMENTUM = 0.9 # Momentum factor for QIFA