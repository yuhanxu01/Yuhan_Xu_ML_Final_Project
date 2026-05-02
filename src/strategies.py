"""
Continual Learning Strategies
=============================
Implements three continual learning strategies to mitigate catastrophic forgetting:

1. Naive Fine-tuning: Sequential learning without anti-forgetting mechanisms
2. Frozen Backbone: Freeze hidden layers, only update classifier
3. Experience Replay: Memory buffer with mixed training

Author: Yuhan Xu
Project: Continual Learning with EWC and MAS
Date: 2026-05-02
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import copy


class ContinualStrategy:
    """
    Base class for continual learning strategies.

    Provides common training functionality and evaluation methods.
    Subclasses should implement the train_task method to define
    their specific continual learning approach.
    """

    def __init__(self, model, device='cpu', learning_rate=0.001, epochs=10):
        """
        Initialize the continual learning strategy.

        Args:
            model: Neural network model (nn.Module)
            device: Device to train on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs per task
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Store training history
        self.history = defaultdict(list)

    def train_epoch(self, train_loader, optimizer, task_id=None):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            optimizer: PyTorch optimizer
            task_id: Current task identifier (for multi-head models)

        Returns:
            float: Average loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            if task_id is not None and hasattr(self.model, 'task_heads'):
                # Multi-head model
                outputs = self.model(batch_x, task_id)
            else:
                # Standard model
                outputs = self.model(batch_x)

            # Compute loss
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(self, test_loader, task_id=None):
        """
        Evaluate model on test data.

        Args:
            test_loader: DataLoader for test data
            task_id: Task identifier (for multi-head models)

        Returns:
            float: Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                if task_id is not None and hasattr(self.model, 'task_heads'):
                    outputs = self.model(batch_x, task_id)
                else:
                    outputs = self.model(batch_x)

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def train_task(self, train_loader, test_loader, task_id):
        """
        Train on a single task. To be implemented by subclasses.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            task_id: Task identifier

        Returns:
            dict: Training statistics
        """
        raise NotImplementedError("Subclasses must implement train_task")

    def get_model_state(self):
        """Get current model state."""
        return {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'history': dict(self.history)
        }


class NaiveFineTuning(ContinualStrategy):
    """
    Naive Fine-tuning: Sequential learning without anti-forgetting mechanisms.

    This is the baseline approach where we simply train on each task sequentially
    without any measures to prevent catastrophic forgetting. This typically shows
    significant forgetting of previous tasks.

    Characteristics:
    - No regularization or constraints
    - All parameters are updated on each task
    - Expected: High catastrophic forgetting
    """

    def train_task(self, train_loader, test_loader, task_id):
        """
        Train on a task without any anti-forgetting measures.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            task_id: Task identifier

        Returns:
            dict: Training statistics
        """
        print(f"\n{'='*60}")
        print(f"Naive Fine-tuning - Training Task {task_id + 1}")
        print(f"{'='*60}")

        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_accuracy = 0.0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, task_id)
            test_acc = self.evaluate(test_loader, task_id)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)

            if test_acc > best_accuracy:
                best_accuracy = test_acc

            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}: "
                      f"Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        print(f"\n   Task {task_id + 1} completed. Best test accuracy: {best_accuracy:.4f}")
        print(f"   [WARNING] No protection against forgetting - expect performance drop on previous tasks!")

        return {
            'task_id': task_id,
            'best_accuracy': best_accuracy,
            'final_train_accuracy': train_acc,
            'final_test_accuracy': test_acc
        }


class FrozenBackbone(ContinualStrategy):
    """
    Frozen Backbone: Freeze hidden layers, only update classifier.

    This strategy freezes the feature extraction layers (hidden layers) after
    the first task and only updates the final classification layer for subsequent tasks.

    Strategy:
    - Task 1: Train entire network normally
    - Task 2+: Freeze fc1 and fc2, only train fc3 (output layer)

    Characteristics:
    - Prevents forgetting in feature extractors
    - Limits adaptation capability for new tasks
    - May underperform if tasks require different features
    """

    def __init__(self, model, device='cpu', learning_rate=0.001, epochs=10):
        super().__init__(model, device, learning_rate, epochs)
        self.frozen_layers = []

    def freeze_backbone(self):
        """Freeze the hidden layers (fc1 and fc2)."""
        if hasattr(self.model, 'fc1'):
            self.model.fc1.requires_grad_(False)
            self.model.fc2.requires_grad_(False)
            self.frozen_layers = ['fc1', 'fc2']
            print("\n   [INFO] Frozen layers: fc1, fc2")
            print(f"   [INFO] Training only: fc3 (output layer)")

    def unfreeze_all(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.frozen_layers = []
        print("\n   [INFO] Unfroze all layers")

    def train_task(self, train_loader, test_loader, task_id):
        """
        Train on a task with frozen backbone strategy.

        For Task 0: Train entire network
        For Task 1+: Freeze hidden layers, train only output layer

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            task_id: Task identifier

        Returns:
            dict: Training statistics
        """
        print(f"\n{'='*60}")
        print(f"Frozen Backbone - Training Task {task_id + 1}")
        print(f"{'='*60}")

        # Apply freezing strategy
        if task_id == 0:
            print("\n   [INFO] Task 1: Training entire network")
            self.unfreeze_all()
        else:
            print(f"\n   [INFO] Task {task_id + 1}: Freezing backbone, training classifier only")
            self.freeze_backbone()

        # Create optimizer (only includes parameters that require gradients)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.learning_rate)

        print(f"   [INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        # Training loop
        best_accuracy = 0.0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, task_id)
            test_acc = self.evaluate(test_loader, task_id)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)

            if test_acc > best_accuracy:
                best_accuracy = test_acc

            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}: "
                      f"Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        print(f"\n   Task {task_id + 1} completed. Best test accuracy: {best_accuracy:.4f}")

        return {
            'task_id': task_id,
            'best_accuracy': best_accuracy,
            'final_train_accuracy': train_acc,
            'final_test_accuracy': test_acc,
            'frozen_layers': self.frozen_layers.copy()
        }


class ExperienceReplay(ContinualStrategy):
    """
    Experience Replay: Memory buffer with mixed training.

    Maintains a small memory buffer of samples from previous tasks.
    When training on a new task, mixes buffer data with current task data
    to retain knowledge of previous tasks.

    Strategy:
    - Store exemplars from each task (default: 100 samples per task)
    - During training, mix buffer data with current task data
    - Balance between new task learning and old task retention

    Characteristics:
    - Requires additional memory for buffer
    - Helps reduce forgetting by replaying old samples
    - Performance depends on buffer size and selection strategy
    """

    def __init__(self, model, device='cpu', learning_rate=0.001, epochs=10,
                 buffer_size=100):
        """
        Initialize Experience Replay strategy.

        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Learning rate
            epochs: Training epochs per task
            buffer_size: Number of samples to store per task
        """
        super().__init__(model, device, learning_rate, epochs)
        self.buffer_size = buffer_size

        # Memory buffer: stores (features, labels) tuples
        self.memory_buffer = []

        print(f"\n   [INFO] Experience Replay initialized")
        print(f"   [INFO] Buffer size per task: {buffer_size} samples")

    def _select_exemplars(self, dataset, task_id, num_samples):
        """
        Select exemplars from a task to add to memory buffer.

        Uses random selection for simplicity. More sophisticated strategies
        (e.g., herding, uncertainty sampling) can be implemented.

        Args:
            dataset: TensorDataset containing features and labels
            task_id: Task identifier
            num_samples: Number of samples to select

        Returns:
            list: Selected (features, labels) tuples
        """
        # Random sampling
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        exemplars = []
        for idx in indices:
            features, label = dataset[idx]
            exemplars.append((features.clone(), label.clone()))

        return exemplars

    def _create_mixed_dataloader(self, current_train_loader, batch_size=64):
        """
        Create a dataloader that mixes current task data with memory buffer.

        Args:
            current_train_loader: DataLoader for current task
            batch_size: Batch size for mixed dataloader

        Returns:
            DataLoader: Mixed dataloader with buffer and current data
        """
        # Get all data from current task
        current_data = []
        current_labels = []

        for batch_x, batch_y in current_train_loader:
            current_data.append(batch_x)
            current_labels.append(batch_y)

        current_data = torch.cat(current_data, dim=0)
        current_labels = torch.cat(current_labels, dim=0)

        # Combine with memory buffer
        if self.memory_buffer:
            buffer_features = []
            buffer_labels = []

            for features, label in self.memory_buffer:
                buffer_features.append(features)
                buffer_labels.append(label)

            buffer_features = torch.stack(buffer_features)
            buffer_labels = torch.stack(buffer_labels)

            # Combine
            mixed_features = torch.cat([current_data, buffer_features], dim=0)
            mixed_labels = torch.cat([current_labels, buffer_labels], dim=0)

            print(f"   [INFO] Mixed dataset: {len(current_data)} current + {len(buffer_features)} buffer = {len(mixed_features)} total")
        else:
            mixed_features = current_data
            mixed_labels = current_labels
            print(f"   [INFO] Using current task data only (buffer empty)")

        # Create mixed dataset and dataloader
        from torch.utils.data import TensorDataset, DataLoader
        mixed_dataset = TensorDataset(mixed_features, mixed_labels)
        mixed_loader = DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True)

        return mixed_loader

    def update_buffer(self, train_loader, task_id):
        """
        Update memory buffer with samples from current task.

        Args:
            train_loader: DataLoader for current task
            task_id: Task identifier
        """
        # Get dataset from dataloader
        dataset = train_loader.dataset

        # Select exemplars
        exemplars = self._select_exemplars(dataset, task_id, self.buffer_size)

        # Add to buffer
        self.memory_buffer.extend(exemplars)

        print(f"   [INFO] Added {len(exemplars)} exemplars from Task {task_id + 1} to buffer")
        print(f"   [INFO] Total buffer size: {len(self.memory_buffer)} samples")

    def train_task(self, train_loader, test_loader, task_id):
        """
        Train on a task with experience replay.

        1. Create mixed dataloader (current task + memory buffer)
        2. Train on mixed data
        3. Update buffer with current task samples

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            task_id: Task identifier

        Returns:
            dict: Training statistics
        """
        print(f"\n{'='*60}")
        print(f"Experience Replay - Training Task {task_id + 1}")
        print(f"{'='*60}")

        # Create mixed dataloader
        mixed_loader = self._create_mixed_dataloader(train_loader, batch_size=64)

        # Create optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_accuracy = 0.0

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(mixed_loader, optimizer, task_id)
            test_acc = self.evaluate(test_loader, task_id)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_acc'].append(test_acc)

            if test_acc > best_accuracy:
                best_accuracy = test_acc

            if (epoch + 1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}: "
                      f"Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

        print(f"\n   Task {task_id + 1} completed. Best test accuracy: {best_accuracy:.4f}")

        # Update buffer with current task samples
        self.update_buffer(train_loader, task_id)

        return {
            'task_id': task_id,
            'best_accuracy': best_accuracy,
            'final_train_accuracy': train_acc,
            'final_test_accuracy': test_acc,
            'buffer_size': len(self.memory_buffer)
        }


def create_strategy(strategy_name, model, device='cpu', learning_rate=0.001,
                    epochs=10, **kwargs):
    """
    Factory function to create continual learning strategies.

    Args:
        strategy_name (str): Name of strategy ('naive', 'frozen', 'replay')
        model: Neural network model
        device: Device to train on
        learning_rate: Learning rate
        epochs: Training epochs
        **kwargs: Additional strategy-specific parameters

    Returns:
        ContinualStrategy: Initialized strategy instance

    Example:
        >>> model = MLP(input_dim=561, hidden_dim=256, output_dim=6)
        >>> strategy = create_strategy('naive', model, device='cpu')
        >>> strategy = create_strategy('replay', model, device='cpu', buffer_size=100)
    """
    strategy_map = {
        'naive': NaiveFineTuning,
        'frozen': FrozenBackbone,
        'replay': ExperienceReplay
    }

    strategy_name = strategy_name.lower()
    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Choose from: {list(strategy_map.keys())}")

    strategy_class = strategy_map[strategy_name]
    return strategy_class(model, device=device, learning_rate=learning_rate,
                         epochs=epochs, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("Continual Learning Strategies - Testing")
    print("="*60)

    from model import MLP

    # Create a simple model
    model = MLP(input_dim=561, hidden_dim=256, output_dim=6)

    # Test strategy creation
    print("\n1. Testing Strategy Factory Function")
    strategies_to_test = ['naive', 'frozen', 'replay']

    for strategy_name in strategies_to_test:
        try:
            if strategy_name == 'replay':
                strategy = create_strategy(strategy_name, model, device='cpu', buffer_size=50)
            else:
                strategy = create_strategy(strategy_name, model, device='cpu')

            print(f"   [OK] Created {strategy_name} strategy: {type(strategy).__name__}")
        except Exception as e:
            print(f"   [ERROR] Failed to create {strategy_name}: {e}")

    print("\n" + "="*60)
    print("Strategy testing completed!")
    print("="*60)
    print("\nStrategies ready for continual learning experiments:")
    print("  - Naive Fine-tuning: Baseline with no protection")
    print("  - Frozen Backbone: Protect feature extractors")
    print("  - Experience Replay: Mix old and new data")
