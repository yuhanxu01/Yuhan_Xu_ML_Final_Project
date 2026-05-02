"""
UCI HAR Dataset Loader for Continual Learning
================================================
This module handles downloading, processing, and splitting the UCI Human Activity Recognition (HAR) dataset
into sequential tasks for continual learning experiments.

Dataset Information:
- Source: UCI Machine Learning Repository
- Features: 561-dimensional sensor measurements
- Classes: 6 activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying)
- Tasks: Split into 3 sequential tasks with 2 classes each

Author: Yuhan Xu
Project: Continual Learning with EWC and MAS
Date: 2026-05-02
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import zipfile
import requests
from tqdm import tqdm


class HARDataset:
    """
    UCI HAR Dataset wrapper for continual learning.

    This class handles downloading, loading, and preprocessing the HAR dataset,
    then splits it into sequential tasks for continual learning experiments.
    """

    def __init__(self, data_dir='./data', url=None, random_seed=42):
        """
        Initialize the HAR dataset handler.

        Args:
            data_dir (str): Directory to store/load the dataset
            url (str): URL to download the dataset from
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.random_seed = random_seed

        # UCI HAR Dataset official URL
        if url is None:
            self.url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
        else:
            self.url = url

        # Set random seed for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Dataset storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # Task definitions: each task contains specific class indices
        # Original HAR labels (1-6) map to activities:
        # 1: Walking, 2: Walking Upstairs, 3: Walking Downstairs, 4: Sitting, 5: Standing, 6: Laying
        self.task_splits = [
            [0, 1],  # Task 1: Walking, Walking Upstairs
            [2, 3],  # Task 2: Walking Downstairs, Sitting
            [4, 5]   # Task 3: Standing, Laying
        ]

    def download_and_extract(self):
        """
        Download the UCI HAR dataset from the official repository and extract it.

        The dataset will be downloaded to self.data_dir and extracted if not already present.
        Shows a progress bar during download.

        Returns:
            str: Path to the extracted dataset directory
        """
        # Check if dataset already exists
        extract_path = os.path.join(self.data_dir, 'UCI HAR Dataset')
        if os.path.exists(extract_path):
            print(f"Dataset already exists at: {extract_path}")
            return extract_path

        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        # Download the dataset
        zip_path = os.path.join(self.data_dir, 'UCI_HAR_Dataset.zip')
        print(f"Downloading UCI HAR Dataset from: {self.url}")

        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        # Extract the zip file
        print(f"\nExtracting dataset to: {self.data_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)

        print(f"Dataset extracted successfully to: {extract_path}")

        # Clean up the zip file
        os.remove(zip_path)
        print("Removed downloaded zip file")

        return extract_path

    def load_data(self):
        """
        Load the HAR dataset from disk.

        This method reads the training and test data from the extracted dataset directory,
        loads features and labels, and applies label remapping from 1-6 to 0-5.

        Returns:
            tuple: (X_train, y_train, X_test, y_test) as numpy arrays
        """
        # Download dataset if not present
        dataset_path = self.download_and_extract()

        # Construct file paths
        train_dir = os.path.join(dataset_path, 'train')
        test_dir = os.path.join(dataset_path, 'test')

        # Load training data
        print("Loading training data...")
        X_train_path = os.path.join(train_dir, 'X_train.txt')
        y_train_path = os.path.join(train_dir, 'y_train.txt')

        self.X_train = np.loadtxt(X_train_path)
        self.y_train = np.loadtxt(y_train_path)

        # Load test data
        print("Loading test data...")
        X_test_path = os.path.join(test_dir, 'X_test.txt')
        y_test_path = os.path.join(test_dir, 'y_test.txt')

        self.X_test = np.loadtxt(X_test_path)
        self.y_test = np.loadtxt(y_test_path)

        # Remap labels from 1-6 to 0-5 for PyTorch compatibility
        print("Remapping labels from [1-6] to [0-5]...")
        self.y_train = self.y_train - 1
        self.y_test = self.y_test - 1

        print(f"Data loaded successfully!")
        print(f"  Training samples: {self.X_train.shape[0]}")
        print(f"  Test samples: {self.X_test.shape[0]}")
        print(f"  Feature dimensions: {self.X_train.shape[1]}")
        print(f"  Number of classes: {len(np.unique(self.y_train))}")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def normalize_data(self):
        """
        Normalize the feature data using StandardScaler.

        Fits the scaler on training data and transforms both training and test data.
        This ensures consistent scaling across train and test sets.
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Normalizing features using StandardScaler...")
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        print("Feature normalization complete.")

    def get_task_data(self, task_id):
        """
        Extract data for a specific task based on class indices.

        Args:
            task_id (int): Task identifier (0, 1, or 2)

        Returns:
            tuple: (X_task_train, y_task_train, X_task_test, y_task_test) for the specified task
        """
        if task_id >= len(self.task_splits):
            raise ValueError(f"Invalid task_id {task_id}. Must be in range [0, {len(self.task_splits)-1}]")

        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get the class indices for this task
        target_classes = self.task_splits[task_id]

        # Filter training data
        train_mask = np.isin(self.y_train, target_classes)
        X_task_train = self.X_train[train_mask]
        y_task_train = self.y_train[train_mask]

        # Filter test data
        test_mask = np.isin(self.y_test, target_classes)
        X_task_test = self.X_test[test_mask]
        y_task_test = self.y_test[test_mask]

        print(f"\nTask {task_id + 1} Data:")
        print(f"  Classes: {target_classes}")
        print(f"  Training samples: {len(X_task_train)}")
        print(f"  Test samples: {len(X_task_test)}")
        print(f"  Class distribution (train): {np.bincount(y_task_train.astype(int))}")

        return X_task_train, y_task_train, X_task_test, y_task_test

    def create_dataloaders(self, X, y, batch_size=64, shuffle=True):
        """
        Create PyTorch DataLoader from numpy arrays.

        Args:
            X (np.ndarray): Feature array
            y (np.ndarray): Label array
            batch_size (int): Batch size for DataLoader
            shuffle (bool): Whether to shuffle the data

        Returns:
            DataLoader: PyTorch DataLoader object
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

    def get_dataloaders(self, batch_size=64, normalize=True):
        """
        Get all task dataloaders for continual learning.

        This is the main entry point for obtaining dataloaders. It loads the data,
        optionally normalizes it, and returns a list of dictionaries containing
        train and test dataloaders for each sequential task.

        Args:
            batch_size (int): Batch size for dataloaders (default: 64)
            normalize (bool): Whether to normalize features (default: True)

        Returns:
            list: List of dictionaries, where each dictionary contains:
                  - 'train_loader': DataLoader for training data of this task
                  - 'test_loader': DataLoader for test data of this task
                  - 'task_id': Task identifier
                  - 'classes': List of class indices in this task

        Example:
            >>> dataset = HARRandomDataset()
            >>> tasks = dataset.get_dataloaders(batch_size=64)
            >>> for task_info in tasks:
            ...     train_loader = task_info['train_loader']
            ...     test_loader = task_info['test_loader']
            ...     # Train on this task...
        """
        # Load data if not already loaded
        if self.X_train is None:
            self.load_data()

        # Normalize features if requested
        if normalize:
            self.normalize_data()

        # Create dataloaders for each task
        task_dataloaders = []

        for task_id in range(len(self.task_splits)):
            print(f"\n{'='*60}")
            print(f"Preparing Task {task_id + 1}/{len(self.task_splits)}")
            print(f"{'='*60}")

            # Get data for this task
            X_train, y_train, X_test, y_test = self.get_task_data(task_id)

            # Create dataloaders
            train_loader = self.create_dataloaders(X_train, y_train, batch_size, shuffle=True)
            test_loader = self.create_dataloaders(X_test, y_test, batch_size, shuffle=False)

            # Store task information
            task_info = {
                'train_loader': train_loader,
                'test_loader': test_loader,
                'task_id': task_id,
                'classes': self.task_splits[task_id]
            }

            task_dataloaders.append(task_info)

        print(f"\n{'='*60}")
        print("Dataset preparation complete!")
        print(f"Total tasks: {len(task_dataloaders)}")
        print(f"{'='*60}\n")

        return task_dataloaders


def get_dataloaders(batch_size=64, data_dir='./data', normalize=True, random_seed=42):
    """
    Convenience function to get HAR dataloaders for continual learning.

    This is a simplified interface for quickly obtaining the task dataloaders
    without needing to instantiate the HARDataset class directly.

    Args:
        batch_size (int): Batch size for dataloaders (default: 64)
        data_dir (str): Directory to store/load the dataset (default: './data')
        normalize (bool): Whether to normalize features (default: True)
        random_seed (int): Random seed for reproducibility (default: 42)

    Returns:
        list: List of dictionaries containing train/test dataloaders for each task

    Example:
        >>> from src.dataset import get_dataloaders
        >>> tasks = get_dataloaders(batch_size=64)
        >>> for i, task in enumerate(tasks):
        ...     print(f"Task {i}: {len(task['train_loader'])} train batches")
    """
    dataset = HARDataset(data_dir=data_dir, random_seed=random_seed)
    return dataset.get_dataloaders(batch_size=batch_size, normalize=normalize)


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("UCI HAR Dataset Loader - Testing")
    print("="*60)

    # Test the dataset loader
    tasks = get_dataloaders(batch_size=64, normalize=True)

    # Display task information
    for task_info in tasks:
        task_id = task_info['task_id']
        train_loader = task_info['train_loader']
        test_loader = task_info['test_loader']
        classes = task_info['classes']

        print(f"\nTask {task_id + 1}:")
        print(f"  Classes: {classes}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Test batch iteration
        for X_batch, y_batch in train_loader:
            print(f"  Batch shape: X={X_batch.shape}, y={y_batch.shape}")
            print(f"  Unique labels in batch: {torch.unique(y_batch).tolist()}")
            break

    print("\n" + "="*60)
    print("Dataset loader test completed successfully!")
    print("="*60)
