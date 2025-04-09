import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Tuple, Dict

class SyntheticCloudDataset(Dataset):
    """Creates a synthetic dataset for binary classification."""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) # CrossEntropyLoss expects long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def generate_synthetic_data(
    num_samples: int, num_features: int, num_classes: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates synthetic classification data."""
    features, labels = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=int(num_features * 0.8), # Make most features informative
        n_redundant=int(num_features * 0.1),
        n_repeated=0,
        n_classes=num_classes,
        n_clusters_per_class=2,
        flip_y=0.05, # Add some noise
        class_sep=1.0, # Adjust separation between classes
        random_state=random_state,
    )
    return features, labels

def load_partitioned_data(
    num_clients: int,
    num_samples_per_client: int,
    num_features: int,
    num_classes: int,
    skewness: float = 0.5, # 0 = IID, 1 = Max Non-IID (only one class per client)
    batch_size: int = 32,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Generates synthetic data and partitions it among clients.
    Introduces Non-IID distribution based on label skewness.
    Returns a list of tuples, where each tuple contains (train_loader, test_loader) for a client.
    """
    np.random.seed(seed)
    client_data_loaders = []

    # Generate a base pool of data
    total_samples = num_samples_per_client * num_clients * 2 # Generate more to allow splitting
    all_features, all_labels = generate_synthetic_data(
        total_samples, num_features, num_classes, seed
    )

    # Sort data by label for easier Non-IID partitioning
    sort_indices = np.argsort(all_labels)
    all_features = all_features[sort_indices]
    all_labels = all_labels[sort_indices]

    # Calculate shards and distribution
    num_shards = num_clients * 2 # Aim for 2 shards per client initially
    shard_size = len(all_labels) // num_shards
    shards_indices = np.arange(num_shards)
    np.random.shuffle(shards_indices) # Shuffle shard order

    client_shard_indices = [[] for _ in range(num_clients)]

    # Distribute shards - this creates the Non-IIDness based on sorted labels
    # Simple approach: give consecutive shards (more skewed) or spread out shards
    # A more controlled way using Dirichlet distribution is common but more complex.
    # Here, we use a simpler label skew based on sorting.
    shards_per_client = num_shards // num_clients
    current_shard = 0
    for i in range(num_clients):
        # Introduce skew: Higher skew means clients get more consecutive shards (similar labels)
        num_consecutive = max(1, int(shards_per_client * skewness))
        num_random = shards_per_client - num_consecutive

        # Assign consecutive shards
        for _ in range(num_consecutive):
            if current_shard < num_shards:
                client_shard_indices[i].append(shards_indices[current_shard])
                current_shard += 1

        # Assign remaining shards randomly from the rest (reduces skew slightly)
        remaining_shards = [s for s in shards_indices if s not in np.concatenate(client_shard_indices)]
        np.random.shuffle(remaining_shards)
        take_count = min(num_random, len(remaining_shards))
        client_shard_indices[i].extend(remaining_shards[:take_count])
        # Remove assigned random shards to avoid reuse (crude way)
        shards_indices = [s for s in shards_indices if s not in remaining_shards[:take_count]]


    # Create datasets and dataloaders for each client
    for i in range(num_clients):
        client_indices = []
        for shard_idx in client_shard_indices[i]:
            start = shard_idx * shard_size
            end = start + shard_size
            client_indices.extend(np.arange(start, end))

        # Ensure we don't exceed requested samples per client after train/test split
        np.random.shuffle(client_indices)
        target_total_samples = num_samples_per_client
        client_indices = client_indices[:int(target_total_samples / 0.8)] # Get slightly more for split

        client_features = all_features[client_indices]
        client_labels = all_labels[client_indices]

        # Split into train/test for this client
        try:
            f_train, f_test, l_train, l_test = train_test_split(
                client_features, client_labels, test_size=0.2, random_state=seed + i, stratify=client_labels
            )
        except ValueError: # Handle cases where a client might get only one class due to extreme skew
             f_train, f_test, l_train, l_test = train_test_split(
                client_features, client_labels, test_size=0.2, random_state=seed + i
            )

        # Ensure we have roughly the target number of samples
        f_train = f_train[:num_samples_per_client]
        l_train = l_train[:num_samples_per_client]
        # Test set size will be proportional

        train_dataset = SyntheticCloudDataset(f_train, l_train)
        test_dataset = SyntheticCloudDataset(f_test, l_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        client_data_loaders.append((train_loader, test_loader))
        print(f"Client {i}: Train samples={len(train_dataset)}, Test samples={len(test_dataset)}, Labels dist: {np.bincount(l_train)}")


    return client_data_loaders


# Example Usage
if __name__ == "__main__":
    NUM_CLIENTS = 3
    SAMPLES_PER_CLIENT = 500
    FEATURES = 10
    CLASSES = 2
    BATCH_SIZE = 16
    SKEW = 0.8 # High skew

    print(f"Generating data for {NUM_CLIENTS} clients with skew {SKEW}...")
    loaders = load_partitioned_data(
        num_clients=NUM_CLIENTS,
        num_samples_per_client=SAMPLES_PER_CLIENT,
        num_features=FEATURES,
        num_classes=CLASSES,
        skewness=SKEW,
        batch_size=BATCH_SIZE
    )

    print(f"\nGenerated {len(loaders)} pairs of dataloaders.")
    # Check first client's data
    train_loader_0, test_loader_0 = loaders[0]
    print(f"Client 0: Train batches={len(train_loader_0)}, Test batches={len(test_loader_0)}")
    features_batch, labels_batch = next(iter(train_loader_0))
    print(f"Client 0: Feature batch shape={features_batch.shape}, Label batch shape={labels_batch.shape}")