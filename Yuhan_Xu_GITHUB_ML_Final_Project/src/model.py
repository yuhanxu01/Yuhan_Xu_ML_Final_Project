"""
Neural Network Model for Continual Learning
============================================
Implements a Multi-Layer Perceptron (MLP) for the UCI HAR dataset classification task.
Designed to support continual learning algorithms (EWC, MAS).

Author: Yuhan Xu
Project: Continual Learning with EWC and MAS
Date: 2026-05-02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for HAR classification.

    Architecture:
    - Input layer: 561 dimensions (UCI HAR features)
    - Hidden layer 1: 256 neurons with ReLU activation
    - Hidden layer 2: 256 neurons with ReLU activation
    - Output layer: 6 neurons (6 activity classes)

    This model is designed for continual learning experiments where:
    - Task boundaries are known
    - Model needs to adapt to new tasks without forgetting old ones
    - Regularization methods (EWC, MAS) will be applied

    Args:
        input_dim (int): Number of input features (default: 561 for UCI HAR)
        hidden_dim (int): Number of neurons in hidden layers (default: 256)
        output_dim (int): Number of output classes (default: 6 for HAR)
        dropout_rate (float): Dropout probability for regularization (default: 0.2)
    """

    def __init__(self, input_dim=561, hidden_dim=256, output_dim=6, dropout_rate=0.2):
        super(MLP, self).__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.

        This helps with gradient flow and training stability,
        especially important for continual learning scenarios.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, task_id=None):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            task_id (int, optional): Task identifier for multi-task learning scenarios.
                                    Currently not used but kept for future extensibility.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim)
        """
        # First hidden layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Second hidden layer with ReLU activation
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer (no activation, logits will be passed to CrossEntropyLoss)
        x = self.fc3(x)

        return x

    def get_predictions(self, x):
        """
        Get class predictions for input data.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Predicted class indices of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_num_parameters(self):
        """
        Get the total number of trainable parameters in the model.

        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        """
        Get a summary of the model architecture.

        Returns:
            dict: Dictionary containing model information
        """
        info = {
            'architecture': 'MLP',
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_parameters': self.get_num_parameters(),
            'layers': [
                f'Linear({self.input_dim} -> {self.hidden_dim}) + ReLU + Dropout',
                f'Linear({self.hidden_dim} -> {self.hidden_dim}) + ReLU + Dropout',
                f'Linear({self.hidden_dim} -> {self.output_dim})'
            ]
        }
        return info


class MultiHeadMLP(nn.Module):
    """
    Multi-Head MLP for continual learning with task-specific output heads.

    This variant uses separate output layers for each task, which can help
    reduce interference between tasks in continual learning settings.

    Architecture:
    - Shared feature extractor: Two hidden layers (256 neurons each)
    - Task-specific heads: One output layer per task (2 classes per task for HAR)

    Args:
        input_dim (int): Number of input features (default: 561)
        hidden_dim (int): Number of neurons in hidden layers (default: 256)
        num_tasks (int): Number of tasks (default: 3 for HAR)
        classes_per_task (int): Number of classes per task (default: 2)
        dropout_rate (float): Dropout probability (default: 0.2)
    """

    def __init__(self, input_dim=561, hidden_dim=256, num_tasks=3, classes_per_task=2, dropout_rate=0.2):
        super(MultiHeadMLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task

        # Shared feature extractor
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Task-specific output heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, classes_per_task) for _ in range(num_tasks)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, task_id):
        """
        Forward pass through the network with task-specific head.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            task_id (int): Task identifier to select the appropriate head

        Returns:
            torch.Tensor: Output logits for the specified task
        """
        # Shared feature extraction
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Task-specific output
        x = self.task_heads[task_id](x)

        return x

    def get_predictions(self, x, task_id):
        """
        Get class predictions for a specific task.

        Args:
            x (torch.Tensor): Input tensor
            task_id (int): Task identifier

        Returns:
            torch.Tensor: Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x, task_id)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_type='standard', input_dim=561, hidden_dim=256, output_dim=6, **kwargs):
    """
    Factory function to create models for continual learning.

    Args:
        model_type (str): Type of model ('standard' or 'multihead')
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension (for standard model)
        **kwargs: Additional arguments passed to model constructor

    Returns:
        nn.Module: Initialized model

    Example:
        >>> model = create_model('standard', input_dim=561, hidden_dim=256)
        >>> multihead_model = create_model('multihead', num_tasks=3, classes_per_task=2)
    """
    if model_type == 'standard':
        return MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)
    elif model_type == 'multihead':
        return MultiHeadMLP(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'standard' or 'multihead'.")


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("MLP Model - Testing")
    print("="*60)

    # Test standard MLP
    print("\n1. Testing Standard MLP...")
    model = MLP(input_dim=561, hidden_dim=256, output_dim=6)

    # Print model info
    info = model.get_model_info()
    print(f"   Architecture: {info['architecture']}")
    print(f"   Parameters: {info['num_parameters']:,}")
    print(f"   Layers: {len(info['layers'])}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 561)
    output = model(x)

    print(f"\n   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: ({batch_size}, 6)")

    # Test predictions
    predictions = model.get_predictions(x)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:5].tolist()}")

    # Test multi-head MLP
    print("\n2. Testing Multi-Head MLP...")
    multihead_model = MultiHeadMLP(input_dim=561, hidden_dim=256, num_tasks=3, classes_per_task=2)

    print(f"   Parameters: {multihead_model.get_num_parameters():,}")

    # Test forward pass for each task
    for task_id in range(3):
        output = multihead_model(x, task_id)
        print(f"   Task {task_id} output shape: {output.shape} (expected: [{batch_size}, 2])")

    print("\n" + "="*60)
    print("Model test completed successfully!")
    print("="*60)
