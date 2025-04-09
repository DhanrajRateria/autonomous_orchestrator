# Autonomous Cloud Security Orchestrator using Federated Learning and Homomorphic Encryption

This project implements a proof-of-concept system demonstrating how Federated Learning (FL) and Homomorphic Encryption (HE) can be combined for privacy-preserving threat detection across distributed cloud environments.

**Core Idea:** Multiple clients (representing different cloud tenants or environments) train a local threat detection model on their private security data. Instead of sharing raw data or plain model parameters, they encrypt their model *updates* using Paillier Homomorphic Encryption (which allows addition on encrypted data) and send them to a central orchestrator (server). The server aggregates these encrypted updates homomorphically, decrypts the *aggregated* result, updates the global model, and sends the improved global model back to the clients.

## Features

*   **Federated Learning:** Uses the [Flower](https://flower.dev/) framework for client-server communication and FL orchestration.
*   **Homomorphic Encryption:** Employs the [Paillier cryptosystem](https://github.com/data61/python-paillier) (`phe` library) for additive homomorphic encryption to protect model updates during aggregation.
*   **Privacy Preservation:** Data remains local to each client. Only encrypted updates are shared. The server only decrypts the final aggregated update, not individual client contributions.
*   **Threat Detection Model:** Includes a simple PyTorch MLP model for binary classification (e.g., Threat vs. No Threat).
*   **Synthetic Data:** Generates synthetic, potentially Non-IID data to simulate client environments for demonstration purposes.
*   **Modular Design:** Code is structured into `client`, `orchestrator`, and `shared` components.
*   **Simulation:** Includes scripts to run the server and multiple clients concurrently (using threads or processes) on a single machine.

## Directory Structure
autonomous_cloud_orchestrator/
├── client/ # Client logic (training, encryption, Flower client)
├── orchestrator/ # Server logic (HE aggregation, decryption, Flower server/strategy)
├── shared/ # Code shared between client/server (model, HE utils, data utils)
├── config/ # Configuration file (config.yaml)
├── data/ # Placeholder for data (generated on the fly)
├── logs/ # Placeholder for logs
├── .gitignore # Git ignore file
├── requirements.txt # Python dependencies
└── README.md # This file


## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd autonomous_cloud_orchestrator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Simulation

1.  **Start the Server:**
    Open a terminal, activate the virtual environment, and run:
    ```bash
    python orchestrator/run_server.py --config config/config.yaml
    ```
    The server will initialize the HE keys (this might take a moment depending on key size) and wait for clients to connect.

2.  **Start the Clients:**
    Open *another* terminal, activate the virtual environment, and run:
    ```bash
    python client/run_clients.py --config config/config.yaml --num_clients 3 --executor thread
    ```
    *   `--num_clients`: Specifies how many clients to simulate (should match or be less than `min_available_clients` in config for training to start).
    *   `--executor`: Use `thread` (lighter, subject to GIL) or `process` (true parallelism, higher overhead, potential pickling issues with complex objects).

    The clients will generate data, initialize, connect to the server, and participate in the federated learning rounds, encrypting their updates. You will see log output from both the server and the clients.

## Configuration

Adjust parameters in `config/config.yaml`:

*   `server_address`: Server host/port. Use `"[::]:8080"` for local simulation accessible only on the same machine, or `0.0.0.0:8080` to be accessible from other machines on the network (use with caution).
*   `num_rounds`: Total federated learning rounds.
*   `num_clients`, `min_fit_clients`, etc.: Control FL participation.
*   `local_epochs`, `batch_size`, `learning_rate`: Client training parameters.
*   `num_features`, `num_hidden_units`, `num_classes`: Model architecture.
*   `num_samples_per_client`, `data_skewness`: Data generation settings.
*   `he_key_length`, `he_precision_bits`: Homomorphic encryption parameters. Larger key length is more secure but *significantly* slower. Higher precision allows finer float representation but increases computational cost.

## How it Works: FL + HE Flow

1.  **Initialization:** Server starts, generates Paillier keypair (Public Key PK, Private Key SK), and waits. Clients start, load data, initialize model.
2.  **Round Start:** Server selects clients for the round.
3.  **Parameter Distribution:** Server sends current global model parameters (plain text) to selected clients.
4.  **Local Training:** Each client updates its local model with the received global parameters. It then trains the model on its local data for `local_epochs`.
5.  **Update Calculation:** Client calculates the *difference* (delta) between its updated local parameters and the parameters it received from the server. `delta = local_params_final - global_params_received`.
6.  **Encryption:** Client *flattens* the delta into a 1D array of floats. It then encodes each float into a large integer and encrypts it using the server's **Public Key (PK)**. This results in a list of Paillier `EncryptedNumber` objects.
7.  **Serialization:** Client serializes this list of encrypted numbers into a byte stream (using `pickle` in this implementation).
8.  **Update Transmission:** Client sends these serialized, encrypted bytes back to the server via Flower (wrapped in a NumPy array as a workaround for Flower's expected format).
9.  **Aggregation (Server):**
    *   Server receives serialized bytes from multiple clients.
    *   Server deserializes bytes back into lists of `EncryptedNumber` objects for each client.
    *   Server performs element-wise **addition directly on the encrypted numbers** (`encrypted_sum = encrypted_delta1 + encrypted_delta2 + ...`). This is the core HE operation.
10. **Decryption (Server):** Server decrypts the *single aggregated sum* using its **Private Key (SK)**. This reveals the sum of all client deltas as large integers. Server decodes these integers back to floats.
11. **Averaging:** Server divides the decrypted sum by the number of contributing clients to get the average delta.
12. **Global Model Update:** Server adds the averaged delta to the previous global model parameters: `new_global_params = old_global_params + average_delta`.
13. **Next Round:** Server sends the `new_global_params` to clients selected for the next round.

## Novelty and Generalization

*   **Novelty:** While combining FL and HE isn't entirely new in research, providing a structured, runnable implementation using standard frameworks like Flower and accessible libraries like `python-paillier` serves as a valuable educational and experimental tool. The specific integration pattern (encrypting deltas, serialization for Flower, custom strategy) is a practical approach.
*   **Generalization:** The core architecture is modular.
    *   **Different Data/Tasks:** Replace `shared/data_utils.py` and `shared/model.py` to adapt to other domains (e.g., medical imaging, financial fraud). The FL/HE logic in `client.py` and `server.py` remains largely the same. For instance, using a CNN for cancer detection on hospital data would involve changing the model and data loader, but the encryption/aggregation flow could be reused.
    *   **Different HE Schemes:** Replace `shared/he_utils.py` with wrappers for more performant libraries (SEAL, PALISADE) or different HE schemes (e.g., BFV/CKKS if multiplication is needed, though aggregation typically only requires addition). This would require adapting the serialization and potentially the strategy logic.
    *   **Cloud Integration:** Real cloud integration would involve replacing `shared/data_utils.py` with connectors to cloud monitoring services (CloudWatch, Azure Monitor, etc.) and potentially deploying clients as containerized services within each cloud environment.

## Limitations and Future Work

*   **HE Performance:** `python-paillier` is extremely slow for practical purposes. Real-world deployments require optimized C++ HE libraries (e.g., Microsoft SEAL via TenSEAL for PyTorch) or hardware acceleration.
*   **Scalability:** The current simulation runs on one machine. Scaling to hundreds or thousands of clients across real networks requires robust infrastructure, optimized communication, and potentially more advanced FL strategies (e.g., hierarchical FL).
*   **Security:** This implementation assumes secure channel communication (handled by Flower/gRPC, potentially with TLS). It doesn't address advanced attacks like model poisoning or inference attacks, which require additional defenses (e.g., differential privacy, robust aggregation rules). Key management is critical in a real deployment.
*   **Model Complexity:** The simple MLP might not be sufficient for complex threat detection.
*   **Serialization Overhead:** Pickling encrypted objects adds overhead. More efficient serialization methods might be needed.
*   **System Programming:** True system-level optimization (e.g., C/C++ bindings for critical paths, async I/O, direct OS interaction) is not implemented but could significantly improve performance.

This codebase provides a foundation for research into privacy-preserving distributed machine learning for security and other sensitive applications.

What the Entire Code Achieves:

This project sets up a simulated Federated Learning system designed for privacy-sensitive tasks like cloud security threat detection (or potentially medical analysis).

Distributed Training: It allows multiple 'clients' (simulating separate cloud environments or hospitals) to collaboratively train a machine learning model without sharing their raw, sensitive data.

Privacy via Federated Learning: The core FL setup ensures data stays local to each client. Only model parameters (or updates) are exchanged.

Enhanced Privacy via Homomorphic Encryption: It adds a strong layer of privacy by having clients encrypt their model updates before sending them to the central server. The server can add these encrypted updates together without decrypting them individually.

Secure Aggregation: The server aggregates the encrypted updates using the additive property of the Paillier homomorphic encryption scheme.

Central Decryption: Only the final, aggregated sum of updates is decrypted by the server using its private key. This prevents the server from seeing any individual client's contribution.

Global Model Improvement: The server uses the decrypted average update to improve a central 'global' model.

Iteration: The improved global model is sent back to clients for further rounds of local training, iteratively enhancing the model's performance on the collective knowledge while maintaining privacy.

Modularity & Generalization: The code is structured so that the core FL + HE mechanism can be potentially reused for different models, datasets, and tasks where distributed, privacy-preserving machine learning is needed.

Demonstration Platform: It provides a runnable (though performance-limited) platform using standard Python libraries (Flower, PyTorch, python-paillier) to understand and experiment with the concepts of combining Federated Learning and Homomorphic Encryption.