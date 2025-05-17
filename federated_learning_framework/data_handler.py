# data_handler.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from PIL import Image
import glob
from pathlib import Path
import torchvision.transforms as transforms # Added for default image transforms

from federated_learning_framework.config import DataConfig, FrameworkConfig # Assuming config.py is in this path

class TabularDataset(Dataset):
    """Dataset for tabular data"""

    def __init__(self, features: np.ndarray, targets: np.ndarray, target_dtype: torch.dtype):
        """
        Initialize dataset with features and targets.
        Args:
            features: Feature matrix as numpy array (float32).
            targets: Target values/classes as numpy array.
            target_dtype: The torch dtype for the target (e.g., long for classification, float for regression).
        """
        if not isinstance(features, np.ndarray) or features.dtype != np.float32:
            raise TypeError("Features must be a float32 numpy array.")
        if not isinstance(targets, np.ndarray):
            raise TypeError("Targets must be a numpy array.")

        self.features = torch.from_numpy(features) # Already float32
        # Convert targets to the specified torch dtype before creating tensor
        self.targets = torch.tensor(targets, dtype=target_dtype)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class ImageDataset(Dataset):
    """Dataset for image data"""

    def __init__(self, image_paths: List[str], targets: List[int], transform=None):
        self.image_paths = image_paths
        self.targets = torch.tensor(targets, dtype=torch.long) # Classification targets are long
        self.transform = transform

        # Define a default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), # Example size, adjust as needed
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
            ])
        self.logger = logging.getLogger("data_handler.image_dataset")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target = self.targets[idx]

        try:
            # Load image using PIL
            image = Image.open(img_path).convert('RGB') # Ensure 3 channels
        except Exception as e:
             self.logger.error(f"Error loading image {img_path}: {e}. Returning dummy data.")
             # Return dummy data of expected shape to avoid crashing DataLoader
             # Shape depends on the transform, get it from a dummy tensor
             dummy_tensor = self.transform(Image.new('RGB', (64, 64))) # Create small dummy image
             return dummy_tensor, target # Return original target

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, target


class DataHandler:
    """
    Handles data loading, preprocessing, and batch creation.
    Manages tabular and image datasets based on configuration and data path.
    """

    def __init__(self, config: FrameworkConfig):
        self.logger = logging.getLogger("data_handler")
        # Store the full config or just the parts needed
        self.full_config = config
        self.data_config = config.data # Keep easy access to data section
        self.task_type = config.model.task_type
        self.system_seed = config.system.seed # Store the system seed

        self.scaler = StandardScaler() if self.data_config.normalize else None
        self.label_encoder = None
        self.original_feature_columns = None
        self.final_feature_columns = None

        # CHANGE: Use data_config here
        if not Path(self.data_config.data_path).exists():
             raise FileNotFoundError(f"Data path specified in config does not exist: {self.data_config.data_path}")

        self.logger.info("Data handler initialized.")

    async def _calculate_pos_weight(self, df: pd.DataFrame, target_column: str) -> Optional[float]:
        """Calculates positive class weight for BCEWithLogitsLoss."""
        if target_column not in df.columns:
            self.logger.error(f"Target column '{target_column}' not in DataFrame for pos_weight calculation.")
            return None
        
        target_series = df[target_column]
        # Ensure target is binarized if not already (e.g. from string labels)
        if not pd.api.types.is_numeric_dtype(target_series) or not all(val in [0,1] for val in target_series.unique() if pd.notna(val)):
            # Try to binarize if it looks like binary classification target
            if len(target_series.unique()) == 2:
                try:
                    # Simple map for binary, assumes 0 is negative, 1 is positive after potential encoding
                    # This part might need adjustment based on how your target is initially encoded
                    # For cancer 'diagnosis', it might be 'M'/'B'. LabelEncoder handles this.
                    # Let's assume target_series for binary is already 0/1 after initial processing steps
                    # like LabelEncoding if it was text.
                    # If it's already numeric 0/1, value_counts will work.
                    pass # Assume it's handled by LabelEncoder or is already 0/1
                except Exception as e:
                    self.logger.warning(f"Could not ensure binary 0/1 target for pos_weight: {e}. Pos_weight may be None.")
                    return None
            else: # Not binary
                return None

        counts = target_series.value_counts()
        if 0 in counts and 1 in counts and counts[1] > 0: # Ensure both classes exist and positive class has samples
            pos_weight_val = counts[0] / counts[1]
            self.logger.info(f"Calculated pos_weight: {pos_weight_val:.4f} (neg_count={counts[0]}, pos_count={counts[1]})")
            return pos_weight_val
        else:
            self.logger.warning(f"Cannot calculate pos_weight. Target distribution: {counts}. Ensure binary 0/1 target with both classes present.")
            return None

    async def load_data(self, data_override_path: Optional[str] = None, val_split: float = 0.2,
                 test_split: float = 0.1) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Loads the dataset specified by data_override_path (for clients) or config.data_path (for server).
        Splits data into train, validation, and test sets.
        Returns DataLoaders for each split. Set test_split=0 for clients usually.
        """
        if data_override_path is None and self.data_config.pos_weight is None and self.task_type == "binary_classification":
            # Load the full dataframe to calculate global pos_weight
            # This logic assumes data_override_path is None for server's initial full data load
            full_data_path_to_load = self.data_config.data_path
            if Path(full_data_path_to_load).is_file() and Path(full_data_path_to_load).suffix == ".csv":
                temp_df_full = pd.read_csv(full_data_path_to_load)
                # --- Preprocess target for pos_weight calculation if needed ---
                # This depends on whether target_column is raw or needs encoding first
                # For breast cancer data, 'diagnosis' is 'M'/'B'.
                if self.data_config.target_column and self.data_config.target_column in temp_df_full.columns:
                    target_for_pos_weight = temp_df_full[self.data_config.target_column]
                    if not pd.api.types.is_numeric_dtype(target_for_pos_weight):
                        le_temp = LabelEncoder() # Use a temporary encoder just for this
                        target_for_pos_weight_encoded = le_temp.fit_transform(target_for_pos_weight)
                        # Create a temporary series/df for calculation
                        temp_df_for_calc = pd.DataFrame({self.data_config.target_column: target_for_pos_weight_encoded})
                        self.full_config.data.pos_weight = await self._calculate_pos_weight(temp_df_for_calc, self.data_config.target_column)
                    else: # Already numeric (0/1)
                        self.full_config.data.pos_weight = await self._calculate_pos_weight(temp_df_full, self.data_config.target_column)
                    
                    if self.full_config.data.pos_weight is not None:
                         self.logger.info(f"Global pos_weight set in config: {self.full_config.data.pos_weight:.4f}")
                else:
                    self.logger.warning("Target column not defined for pos_weight calculation during initial server data load.")
            else:
                self.logger.warning(f"Could not load full dataset from {full_data_path_to_load} to calculate pos_weight.")
        data_path = data_override_path if data_override_path else self.data_config.data_path
        self.logger.info(f"Attempting to load data from: {data_path}")

        if not Path(data_path).exists():
            raise ValueError(f"Data path does not exist: {data_path}")

        # Determine dataset type
        path_obj = Path(data_path)
        if path_obj.is_file() and path_obj.suffix == ".csv":
            dataset = self._load_tabular_data(str(path_obj))
            data_type = "tabular"
        elif path_obj.is_dir():
            # Assuming image data directory structure (root/class/image.ext)
             dataset = self._load_image_data(str(path_obj))
             data_type = "image"
             # For images, input/output shapes are often fixed by the model (e.g., CNN)
             # We might need to update config based on loaded image data info if needed.
             # Example: self.data_config.output_shape = [len(dataset.classes)] if hasattr(dataset, 'classes') else [1]
        else:
            raise ValueError(f"Unsupported data format or path: {data_path}")

        if dataset is None:
             raise RuntimeError(f"Failed to load dataset from {data_path}")

        # Split dataset
        total_size = len(dataset)
        if total_size == 0:
             raise ValueError(f"Loaded dataset from {data_path} is empty.")

        # Ensure splits are valid
        if not (0 <= val_split < 1 and 0 <= test_split < 1 and val_split + test_split < 1):
             raise ValueError(f"Invalid split sizes: val={val_split}, test={test_split}. Must be >=0 and sum < 1.")

        test_size = int(total_size * test_split)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size - test_size

        if train_size <= 0:
             self.logger.warning(f"Train size is {train_size} after splitting. Check split ratios or dataset size.")
             # Handle edge case: assign at least one sample if possible, or raise error
             if total_size > val_size + test_size:
                  train_size = 1
                  # Adjust val or test slightly if needed
                  if val_size > 0 : val_size -=1
                  elif test_size > 0 : test_size -=1
                  else: raise ValueError("Cannot create non-empty train split.")
             else:
                  raise ValueError("Dataset too small for specified validation/test splits, resulting in zero training samples.")


        self.logger.info(f"Splitting data: Total={total_size}, Train={train_size}, Val={val_size}, Test={test_size}")

        # Perform the split
        # Use a fixed generator for reproducibility if needed (requires seed)
        generator = torch.Generator().manual_seed(self.system_seed)
        try:
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size], generator=generator # Pass generator
            )
        except ValueError as e:
             self.logger.error(f"Error during random_split (total={total_size}, splits=[{train_size}, {val_size}, {test_size}]): {e}", exc_info=True)
             raise

        # --- Create DataLoaders ---
        # Use num_workers from system config
        num_workers = self.full_config.system.num_workers
        batch_size = self.data_config.batch_size
        persistent_workers = num_workers > 0 # Only use persistent workers if num_workers > 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False, # Good practice with workers
            persistent_workers=persistent_workers,
            drop_last=True # Drop last incomplete batch for consistent batch sizes during training
        )

        val_loader = None
        if val_size > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.data_config.batch_size * 2, # Often larger batch size for validation
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if num_workers > 0 else False,
                persistent_workers=persistent_workers
            )
        else:
             self.logger.info("No validation set created (val_split=0 or dataset too small).")


        test_loader = None
        if test_size > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.data_config.batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if num_workers > 0 else False,
                persistent_workers=persistent_workers
            )
        else:
             self.logger.info("No test set created (test_split=0 or dataset too small).")

        self.logger.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}, Test batches: {len(test_loader) if test_loader else 0}")

        return train_loader, val_loader, test_loader


    def _load_tabular_data(self, data_path: str) -> Optional[TabularDataset]:
        """Loads and preprocesses tabular data from a CSV file."""
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns from {data_path}")
            if df.empty:
                 self.logger.error("Loaded DataFrame is empty.")
                 return None
            
            cols_to_drop = []
            if 'Unnamed: 32' in df.columns:
                cols_to_drop.append('Unnamed: 32')

            if cols_to_drop:
                self.logger.warning(f"Dropping identified problematic columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)

            # --- Identify Features and Target ---
            if self.data_config.target_column and self.data_config.target_column in df.columns:
                target_col = self.data_config.target_column
            else:
                # Try to infer target (e.g., last column)
                potential_targets = ['target', 'label', 'class', 'diagnosis', 'y']
                target_col = df.columns[-1] # Default to last
                for pt in potential_targets:
                    if pt in df.columns:
                        target_col = pt
                        break
                self.logger.warning(f"Target column not specified or found, inferring '{target_col}' as target.")
                # Update config if target was inferred - do this carefully if clients use same config
                # self.data_config.target_column = target_col # Might be better done once in generator

            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame after potential drops.")


            # --- Determine Feature Columns to Use ---
            # Start with columns specified in config, if available
            if self.data_config.feature_columns:
                # Filter config features: must exist in df and not be the target or dropped columns
                feature_cols = [
                    col for col in self.data_config.feature_columns
                    if col in df.columns and col != target_col
                ]
                # Log if any specified features were missing or dropped
                original_config_features = set(self.data_config.feature_columns)
                final_used_features = set(feature_cols)
                missing_or_dropped = original_config_features - final_used_features - {target_col} # Exclude target
                if missing_or_dropped:
                    self.logger.warning(f"Specified feature columns not used (missing or dropped): {list(missing_or_dropped)}")
            else:
                # If no features specified in config, use all remaining columns except target
                feature_cols = [col for col in df.columns if col != target_col]

            if not feature_cols:
                raise ValueError("No usable feature columns identified after filtering.")

            self.logger.info(f"Using {len(feature_cols)} feature columns: {feature_cols[:10]}...") # Log first few
            self.original_feature_columns = list(feature_cols) # Store names of columns *used* for processing

            # --- Preprocess Features ---
            X_list = [] # Collect processed feature columns/groups
            final_feature_names = []

            for col in feature_cols:
                column_data = df[col]
                if pd.api.types.is_numeric_dtype(column_data):
                    # Handle NaNs in numeric columns
                    if column_data.isnull().any():
                        mean_val = column_data.mean()
                        self.logger.warning(f"Numeric column '{col}' has NaNs, filling with mean ({mean_val:.4f}).")
                        column_data = column_data.fillna(mean_val)
                    X_list.append(column_data.values.reshape(-1, 1))
                    final_feature_names.append(col)
                elif pd.api.types.is_categorical_dtype(column_data) or column_data.dtype == 'object':
                     # Handle NaNs in categorical columns (e.g., fill with mode or a placeholder)
                    if column_data.isnull().any():
                         mode_val = column_data.mode()[0] if not column_data.mode().empty else "Unknown"
                         self.logger.warning(f"Categorical column '{col}' has NaNs, filling with mode ('{mode_val}').")
                         column_data = column_data.fillna(mode_val)

                    # One-Hot Encode categorical features
                    self.logger.info(f"One-hot encoding categorical column '{col}'.")
                    dummies = pd.get_dummies(column_data, prefix=col, dummy_na=False) # dummy_na=False as we filled NaNs
                    X_list.append(dummies.values)
                    final_feature_names.extend(dummies.columns.tolist())
                else:
                     self.logger.warning(f"Column '{col}' has unsupported dtype {column_data.dtype}, attempting to convert to numeric.")
                     try:
                          column_data_numeric = pd.to_numeric(column_data, errors='coerce')
                          if column_data_numeric.isnull().any():
                               mean_val = column_data_numeric.mean()
                               self.logger.warning(f"Converted column '{col}' has NaNs, filling with mean ({mean_val:.4f}).")
                               column_data_numeric = column_data_numeric.fillna(mean_val)
                          X_list.append(column_data_numeric.values.reshape(-1, 1))
                          final_feature_names.append(col)
                     except Exception as e:
                          self.logger.error(f"Could not convert column '{col}' to numeric. Skipping column. Error: {e}")

            if not X_list:
                 raise ValueError("No features remaining after preprocessing.")

            # Combine processed features
            X = np.concatenate(X_list, axis=1).astype(np.float32)
            self.final_feature_columns = final_feature_names
            self.logger.info(f"Final feature matrix shape: {X.shape}")


            # --- Preprocess Target ---
            y_raw = df[target_col]
            target_dtype: torch.dtype

            if self.task_type == "regression":
                self.logger.info(f"Processing target '{target_col}' for regression.")
                if y_raw.isnull().any():
                    raise ValueError(f"Target column '{target_col}' for regression contains NaN values.")
                try:
                    y = y_raw.values.astype(np.float32)
                except ValueError:
                     raise ValueError(f"Target column '{target_col}' cannot be converted to float for regression.")
                target_dtype = torch.float32
                # Update output shape if needed (regression usually has 1 output)
                if self.data_config.output_shape != [1]:
                     self.logger.warning(f"Task is regression, setting output_shape to [1] from {self.data_config.output_shape}")
                     self.data_config.output_shape = [1]

            elif self.task_type in ["classification", "binary_classification"]:
                self.logger.info(f"Processing target '{target_col}' for classification.")
                 # Handle potential NaNs in target for classification (usually drop or error)
                if y_raw.isnull().any():
                    raise ValueError(f"Target column '{target_col}' for classification contains NaN values.")

                # Use LabelEncoder for consistent mapping
                if self.label_encoder is None:
                     self.label_encoder = LabelEncoder()
                     y = self.label_encoder.fit_transform(y_raw)
                     self.logger.info(f"Fitted LabelEncoder. Classes: {self.label_encoder.classes_}")
                else:
                     # Use existing encoder (important for consistency across client datasets)
                     try:
                          y = self.label_encoder.transform(y_raw)
                          self.logger.info("Applied existing LabelEncoder.")
                     except ValueError as e:
                          # Handle unseen labels during transform (might happen with non-iid data)
                          self.logger.error(f"LabelEncoder error: {e}. This might indicate unseen labels in this data partition.")
                          # Option 1: Raise error (strict)
                          # raise ValueError(f"Unseen labels encountered in target column '{target_col}'. Ensure all possible labels were seen during initial fit or handle appropriately.") from e
                          # Option 2: Map unseen to a specific category (e.g., -1 or len(classes)) - requires careful handling in model/loss
                          # Option 3: Skip rows with unseen labels (might reduce data)
                          # Let's raise error for now, as it indicates a potential issue with data splitting or understanding.
                          raise

                num_classes = len(self.label_encoder.classes_)
                self.logger.info(f"Target distribution: {np.unique(y, return_counts=True)}")

                if self.task_type == "binary_classification":
                    if num_classes != 2:
                        raise ValueError(f"Task is 'binary_classification' but found {num_classes} unique classes in target '{target_col}'. Labels: {self.label_encoder.classes_}")
                    # Target for BCEWithLogitsLoss should be float32 (0.0 or 1.0)
                    target_dtype = torch.float32
                    y = y.astype(np.float32) # Convert 0, 1 integers to floats
                    # Update output shape if needed (binary usually has 1 output neuron)
                    if self.data_config.output_shape != [1]:
                         self.logger.warning(f"Task is binary classification, setting output_shape to [1] from {self.data_config.output_shape}")
                         self.data_config.output_shape = [1]
                else: # Multi-class classification
                    # Target for CrossEntropyLoss should be long (class indices)
                    target_dtype = torch.long
                    y = y.astype(np.int64) # Ensure integer type for indices
                    # Update output shape if needed
                    if self.data_config.output_shape != [num_classes]:
                         self.logger.warning(f"Task is classification, setting output_shape to [{num_classes}] from {self.data_config.output_shape}")
                         self.data_config.output_shape = [num_classes]

            else:
                raise ValueError(f"Unsupported task_type: {self.task_type}")


            # --- Normalize Features ---
            if self.data_config.normalize and self.scaler:
                if not hasattr(self.scaler, "scale_"): # Check if fitted using `scale_` attribute
                    self.logger.info("Fitting StandardScaler on features.")
                    # Add Robust Scaling: Handle zero variance columns
                    try:
                        # Use fit_transform only if not fitted
                        X = self.scaler.fit_transform(X)
                        # Check for NaNs *after* scaling
                        if np.isnan(X).any():
                            self.logger.warning("NaN values detected *after* scaling. Check input data variance. Attempting to impute NaNs with 0.")
                            X = np.nan_to_num(X, nan=0.0) # Replace NaNs with 0 after scaling
                    except ValueError as e:
                        self.logger.error(f"Error during StandardScaler fit_transform: {e}. Check for columns with zero variance.")
                        # Fallback: Skip normalization? Or impute before scaling?
                        # For now, we try nan_to_num after potential error.
                        X = np.nan_to_num(X, nan=0.0)
                else:
                    self.logger.info("Applying existing StandardScaler.")
                    # Add Robust Scaling during transform too
                    try:
                        X = self.scaler.transform(X)
                        if np.isnan(X).any():
                            self.logger.warning("NaN values detected *after* scaling transform. Imputing with 0.")
                            X = np.nan_to_num(X, nan=0.0)
                    except ValueError as e:
                        self.logger.error(f"Error during StandardScaler transform: {e}.")
                        X = np.nan_to_num(X, nan=0.0)



            # --- Update Config Shapes ---
            final_input_dim = X.shape[1]
            if self.data_config.input_shape != [final_input_dim]:
                self.logger.warning(f"Input shape mismatch after preprocessing. Updating config input_shape from {self.data_config.input_shape} to [{final_input_dim}]")
                self.data_config.input_shape = [final_input_dim]

            # --- Create Dataset ---
            self.logger.info(f"Creating TabularDataset with feature shape {X.shape} and target shape {y.shape}, target dtype {target_dtype}")
            dataset = TabularDataset(X, y, target_dtype)
            return dataset

        except Exception as e:
            self.logger.error(f"Error loading or processing tabular data from {data_path}: {e}", exc_info=True)
            return None


    def _load_image_data(self, data_path: str) -> Optional[ImageDataset]:
        """Loads image data assuming 'data_path/class_name/image.jpg' structure."""
        self.logger.info(f"Loading image data from directory: {data_path}")
        try:
            root_dir = Path(data_path)
            classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
            if not classes:
                 raise ValueError(f"No class subdirectories found in {data_path}")

            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            self.logger.info(f"Found {len(classes)} classes: {classes}")

            image_paths = []
            targets = []
            supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]

            for cls_name in classes:
                class_idx = class_to_idx[cls_name]
                class_dir = root_dir / cls_name
                img_count = 0
                for ext in supported_extensions:
                    for img_path in class_dir.glob(ext):
                        image_paths.append(str(img_path))
                        targets.append(class_idx)
                        img_count += 1
                self.logger.debug(f"Found {img_count} images for class '{cls_name}'")

            if not image_paths:
                 raise ValueError(f"No images found in class subdirectories of {data_path}")

            self.logger.info(f"Found {len(image_paths)} total images across {len(classes)} classes.")

            # Determine input/output shapes from data
            # Output shape is number of classes
            self.data_config.output_shape = [len(classes)]
            # Input shape depends on transforms (e.g., ToTensor + Normalize -> [C, H, W])
            # We might need to pass a sample image through transforms to know the exact shape
            # Or rely on a standard shape defined in the model config / task context
            # For now, let's assume a default or rely on config being correct
            # Example: Get shape after default transform
            if ImageDataset.default_transform:
                 try:
                      sample_img_path = image_paths[0]
                      sample_img = Image.open(sample_img_path).convert('RGB')
                      sample_tensor = ImageDataset.default_transform(sample_img)
                      self.data_config.input_shape = list(sample_tensor.shape)
                      self.logger.info(f"Inferred image input shape from default transform: {self.data_config.input_shape}")
                 except Exception as e:
                      self.logger.warning(f"Could not infer image input shape: {e}. Using config value: {self.data_config.input_shape}")


            # Create dataset (transforms are handled within ImageDataset)
            dataset = ImageDataset(image_paths, targets) # Use default transform for now
            setattr(dataset, 'classes', classes) # Store classes list in dataset object

            return dataset

        except Exception as e:
            self.logger.error(f"Error loading image data from {data_path}: {e}", exc_info=True)
            return None

    def get_feature_names(self) -> Optional[List[str]]:
        """Returns the final list of feature names after preprocessing."""
        return self.final_feature_columns