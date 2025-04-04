import random # For CLIENT_ID generation

# client/config.py
SERVER_URL = 'http://127.0.0.1:5000'
CLIENT_ID = f"cloud_env_{random.randint(1000, 9999)}" # Simulate unique client ID per instance

# Data Simulation Config
NUM_SAMPLES = 500 # Number of log entries/events per client per round
FEATURE_COUNT = 10 # MUST MATCH SERVER'S model_manager.MODEL_FEATURE_COUNT
THREAT_RATIO = 0.1 # Percentage of simulated data that represents threats (can vary per client)

# Training Config
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01 # Relevant if using SGD-based models, less so for LogisticRegression default solver

PRECISION_FACTOR = 10**6 # MUST MATCH SERVER'S config.PRECISION_FACTOR