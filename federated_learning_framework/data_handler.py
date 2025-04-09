import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from PIL import Image
import glob
from pathlib import Path

from federated_learning_framework.config import DataConfig

class TabularDataset(Dataset):
    """Dataset for tabular data (e.g., cancer detection)"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long if targets.ndim == 1 else torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ImageDataset(Dataset):
    """Dataset for image data"""
    
    def __init__(self, image_paths: List[str], targets: List[int], transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target = self.targets[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = torch.tensor(np.array(image).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
        
        return image, torch.tensor(target, dtype=torch.long)

class DataHandler:
    """
    Handles data loading, preprocessing and batch creation for federated learning.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data handler.
        
        Args:
            config: Data configuration
        """
        self.logger = logging.getLogger("data_handler")
        self.config = config
        
        # Preprocessing components
        self.scaler = StandardScaler() if config.normalize else None
        
        self.logger.info("Data handler initialized")
    
    def load_data(self, val_split: float = 0.2, 
                 test_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load dataset and create data loaders.
        
        Args:
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        data_path = self.config.data_path
        if not data_path or not os.path.exists(data_path):
            raise ValueError(f"Data path not found: {data_path}")
        
        # Determine dataset type from file extension
        if data_path.endswith(".csv"):
            dataset = self._load_tabular_data(data_path)
        elif os.path.isdir(data_path):
            dataset = self._load_image_data(data_path)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
        
        # Split dataset into train, validation and test sets
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size - test_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        self.logger.info(f"Data loaded: {train_size} training, {val_size} validation, {test_size} test samples")
        
        return train_loader, val_loader, test_loader
    
    def _load_tabular_data(self, data_path: str) -> Dataset:
        """Load tabular data from CSV file"""
        df = pd.read_csv(data_path)
        
        # Process features and target
        if self.config.target_column:
            target_col = self.config.target_column
            feature_cols = [col for col in df.columns if col != target_col]
        else:
            # Assume last column is target
            feature_cols = list(df.columns[:-1])
            target_col = df.columns[-1]
        
        # Override with configured feature columns if provided
        if self.config.feature_columns:
            feature_cols = self.config.feature_columns
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Apply normalization if enabled
        if self.config.normalize and self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Create dataset
        dataset = TabularDataset(X, y)
        
        self.logger.info(f"Loaded tabular data from {data_path}: {X.shape[0]} samples, {X.shape[1]} features")
        
        return dataset
    
    def _load_image_data(self, data_path: str) -> Dataset:
        """Load image data from directory"""
        # Assume subdirectories are class names
        classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        classes.sort()
        
        # Create class to index mapping
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        # Collect image paths and targets
        image_paths = []
        targets = []
        
        for cls in classes:
            cls_path = os.path.join(data_path, cls)
            for img_path in glob.glob(os.path.join(cls_path, "*.jpg")) + \
                          glob.glob(os.path.join(cls_path, "*.jpeg")) + \
                          glob.glob(os.path.join(cls_path, "*.png")):
                image_paths.append(img_path)
                targets.append(class_to_idx[cls])
        
        # Create dataset
        dataset = ImageDataset(image_paths, targets)
        
        self.logger.info(f"Loaded image data from {data_path}: {len(dataset)} samples, {len(classes)} classes")
        
        return dataset