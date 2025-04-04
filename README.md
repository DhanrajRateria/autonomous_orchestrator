# Autonomous Cloud Security Orchestrator using FL and HE

This project demonstrates a conceptual Autonomous Cloud Security Orchestrator. It uses Federated Learning (FL) to train a threat detection model across multiple simulated cloud environments without sharing raw data. Homomorphic Encryption (HE) (specifically, the Paillier cryptosystem) is employed to protect the confidentiality of model updates during aggregation. The system simulates autonomous actions based on the learned global model's insights.

## Architecture

*   **Orchestrator Server (`server/`):**
    *   Manages the overall FL process (rounds).
    *   Initializes and distributes the global threat detection model (Logistic Regression).
    *   Generates and manages Paillier HE keys (public/private).
    *   Receives encrypted model updates (weight differences) from clients.
    *   Uses Paillier's additive homomorphism to aggregate encrypted updates.
    *   Decrypts the aggregated update using the private key.
    *   Updates the global model using the averaged decrypted updates.
    *   Simulates autonomous actions (e.g., logging a warning) if threat metrics exceed a threshold.
    *   Uses Flask for basic API communication.
    *   Uses `concurrent.futures` to simulate efficient handling of cryptographic operations.
*   **Cloud Client Agent (`client/`):**
    *   Simulates an agent running in a specific cloud environment.
    *   Registers with the server to get the HE public key and model details.
    *   Generates synthetic local security data (simulating logs/events) with potential biases.
    *   Receives the global model from the server.
    *   Trains the model locally on its data (`local_trainer.py`).
    *   Calculates the difference between its updated weights and the received global weights.
    *   Encrypts this weight difference vector using the server's public HE key (`crypto_manager.py`).
    *   Sends the encrypted update back to the server.

## Core Technologies

*   **Federated Learning:** Enables collaborative model training without centralizing sensitive data. The server orchestrates rounds, and clients train locally.
*   **Homomorphic Encryption (Paillier):** Allows the server to perform computations (addition for aggregation) on encrypted data without decrypting individual contributions. This protects the confidentiality of each client's model update.
*   **System Programming Concepts (Simulated):** The architecture emphasizes component separation (server, client, crypto, model). Concurrency (`ThreadPoolExecutor`) is used on the server to handle potentially intensive cryptographic operations efficiently, mimicking optimizations needed in real systems. Efficient libraries like NumPy are used.
*   **Autonomous Orchestration:** The server includes a placeholder `evaluate_model_and_trigger_action` function that simulates taking automated security actions based on the global model's state (e.g., detecting a high probability of threats).

## Setup and Running

**1. Prerequisites:**
   *   Python 3.8+
   *   Install GMP library (see instructions in the code comments or original prompt).
        *   Linux (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install libgmp-dev`
        *   Linux (Fedora/CentOS): `sudo yum install gmp-devel`
        *   macOS (Homebrew): `brew install gmp`
        *   Windows: Use WSL or find pre-compiled binaries/conda.

**2. Install Python Dependencies:**
   Navigate to the `server` directory and run:
   ```bash
   pip install -r requirements.txt
```
Navigate to the client directory and run:
```bash
pip install -r requirements.txt
```

**3. Run the Orchestrator Server:**
Open a terminal in the autonomous_cloud_orchestrator root directory.
```bash
python -m server.main
```
The server will initialize, generate HE keys, start the Flask API, and begin the federated learning rounds.

**4. Run Client Agents:**
Open separate terminals for each client you want to simulate (at least MIN_CLIENTS_FOR_AGGREGATION as defined in server/config.py, default is 2). In each client terminal, navigate to the autonomous_cloud_orchestrator root directory.

```bash
# Terminal 1 (Client 1)
python -m client.main

# Terminal 2 (Client 2)
python -m client.main
```
Each client will generate a unique ID, register with the server, and participate in the learning rounds by fetching the model, training locally, encrypting updates, and submitting them.

**5. Observe:**
Watch the logs in the server and client terminals. You'll see:
- Server starting rounds.
- Clients registering and fetching the model.
- Clients generating data and training.
- Clients encrypting and submitting updates.
- Server receiving updates, aggregating, decrypting.
- Server updating the global model.
- Server potentially triggering simulated autonomous actions if the threat threshold is met.

## Novelty and Modifications
**Combination:** The specific combination of FL for distributed cloud security and HE (Paillier) for update privacy with an autonomous orchestration layer is relatively novel in practical, open-source examples.

**System Aspects:** Emphasis on component interaction and simulated concurrency addresses system-level design.

**Autonomous Trigger:** Explicitly including a simulated autonomous action based on model evaluation adds to the orchestration concept.

**Modifications from Standard FL:** Incorporates the non-trivial steps of HE encryption/decryption and the necessary float-to-int scaling (PRECISION_FACTOR) required for Paillier.

## Limitations & Future Work
**Simulation:** Uses synthetic data and simulates cloud environments locally. Real-world integration requires cloud APIs, actual log ingestion, and robust agent deployment.

**HE Performance:** Paillier encryption/decryption, especially with larger keys (required for security), is computationally intensive. Performance optimization (hardware acceleration, optimized libraries) is crucial.

**Precision Loss:** Converting floats to integers for HE can cause precision loss. The PRECISION_FACTOR needs careful tuning.

**Security:** This is a conceptual demo. Production systems need TLS/SSL, client authentication, potentially Differential Privacy, robust key management, and protection against model poisoning attacks.

**Model:** Uses a simple Logistic Regression model. More complex threats require more sophisticated models (e.g., LSTMs for sequential logs, GNNs for network graphs, Transformers).

**Scalability:** The simple Flask server and in-memory state need replacement with scalable infrastructure (e.g., message queues, databases, distributed task frameworks) for many clients.

**Error Handling:** Basic error handling is included; production systems require more resilience.

---

**What the Entire Code Achieves:**

This project implements a proof-of-concept **Autonomous Cloud Security Orchestrator**. Its core purpose is to **improve threat detection capabilities across multiple, isolated cloud environments without compromising the privacy of local security data**.

Here's a breakdown of its achievement:

1.  **Distributed Learning:** It leverages **Federated Learning (FL)**, allowing a central orchestrator to build a powerful, global threat detection model by learning from diverse data residing in separate client environments (simulated clouds). Crucially, raw security logs or sensitive local data never leave the client environment.
2.  **Privacy Preservation:** It employs **Homomorphic Encryption (HE)**, specifically the additively homomorphic Paillier cryptosystem. Clients encrypt their model *updates* (not raw data) before sending them. The server can *aggregate* these encrypted updates directly. Only the *final aggregated result* is decrypted, preventing the server (or an eavesdropper) from learning any individual client's specific contribution.
3.  **Centralized Orchestration:** The `server` component acts as the central **orchestrator**. It manages the entire process: distributing the model, coordinating learning rounds, securely aggregating updates, refining the global model, and importantly, making decisions based on the model's output.
4.  **Enhanced Threat Detection:** By combining insights from multiple environments, the global model becomes more robust and generalizable than any model trained on a single environment's data. It can potentially identify widespread or subtle attack patterns that might be missed locally.
5.  **Autonomous Response (Simulated):** It demonstrates the concept of **autonomy**. Based on the insights from the collaboratively trained global model (e.g., predicting a high likelihood of threats based on certain patterns), the orchestrator can automatically trigger security actions (simulated here by logging warnings, but designed to integrate with real security tools/APIs).
6.  **System Design Principles:** Although using high-level Python, the structure (client-server, distinct managers for crypto/model, use of concurrency) reflects **system programming** principles necessary for building efficient, maintainable, and scalable distributed security systems.

In essence, the code provides a working, albeit simplified, blueprint for a next-generation security system that is collaborative, privacy-preserving, and capable of automated response, tailored for the complexities of multi-cloud or distributed enterprise environments.