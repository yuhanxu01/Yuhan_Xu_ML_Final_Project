"""
Evaluation Metrics for Continual Learning
==========================================
Implements evaluation metrics for measuring catastrophic forgetting and
overall performance in continual learning scenarios.

Author: Yuhan Xu
Project: Continual Learning with EWC and MAS
Date: 2026-05-02

References:
- Lopez-Paz & Ranzato (2017): "Gradient Episodic Memory for Continual Learning"
- Chaudhry et al. (2018): "Elastic Weight Consolidation"
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class ContinualLearningMetrics:
    """
    Computes and tracks continual learning metrics over training.

    This class calculates Average Accuracy (AA) and Backward Transfer (BWT)
    to evaluate model performance in continual learning scenarios.
    """

    def __init__(self, num_tasks: int):
        """
        Initialize metrics tracker.

        Args:
            num_tasks (int): Total number of tasks in the continual learning setup
        """
        self.num_tasks = num_tasks

        # Matrix to store accuracy: shape (num_tasks, num_tasks)
        # accuracy_matrix[i, j] = accuracy on task j after learning task i
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))

        # Track after which task we're evaluating
        self.current_task = 0

    def evaluate_task(self, model, task_id: int, test_loader, device='cpu') -> float:
        """
        Evaluate model on a specific task.

        Args:
            model: The neural network model
            task_id (int): Task identifier to evaluate
            test_loader: DataLoader for test data
            device: Device to run evaluation on

        Returns:
            float: Accuracy on this task (0.0 to 1.0)
        """
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # Get predictions based on model type
                if hasattr(model, 'task_heads'):  # Multi-head model
                    predictions = model.get_predictions(batch_x, task_id)
                else:  # Standard model
                    # For standard model, we evaluate on all 6 classes
                    # but only compute accuracy for the classes in this task
                    logits = model(batch_x)
                    predictions = torch.argmax(logits, dim=1)

                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def update_metrics(self, model, task_loaders: List[Dict], device='cpu'):
        """
        Evaluate model on all tasks seen so far and update metrics.

        This should be called after training on each new task.

        Args:
            model: The trained model
            task_loaders (list): List of task dictionaries containing test loaders
            device: Device to run evaluation on

        Returns:
            dict: Current metrics including AA and partial BWT
        """
        # Evaluate on all tasks up to current_task
        for task_id in range(self.current_task + 1):
            test_loader = task_loaders[task_id]['test_loader']
            accuracy = self.evaluate_task(model, task_id, test_loader, device)
            self.accuracy_matrix[self.current_task, task_id] = accuracy

        # Compute metrics
        aa = self.compute_average_accuracy()
        bwt = self.compute_backward_transfer()

        metrics = {
            'current_task': self.current_task,
            'average_accuracy': aa,
            'backward_transfer': bwt,
            'accuracy_matrix': self.accuracy_matrix[:self.current_task + 1, :self.current_task + 1].copy()
        }

        # Move to next task
        self.current_task += 1

        return metrics

    def compute_average_accuracy(self) -> float:
        """
        Compute Average Accuracy (AA) after learning current task.

        AA is the average accuracy over all tasks seen so far.
        Higher AA indicates better overall performance without forgetting.

        Formula:
            AA_t = (1/t) * Σ_{j=1}^{t} R_{i,j}

        where R_{i,j} is accuracy on task j after learning task i (i >= j)

        Returns:
            float: Average accuracy over all seen tasks (0.0 to 1.0)
        """
        if self.current_task == 0:
            return self.accuracy_matrix[0, 0]

        # Average over all tasks seen so far (diagonal and below)
        seen_tasks = self.current_task + 1
        accuracies = []

        for task_id in range(seen_tasks):
            # Get accuracy on task_id after learning the most recent task
            acc = self.accuracy_matrix[self.current_task, task_id]
            accuracies.append(acc)

        return np.mean(accuracies)

    def compute_backward_transfer(self) -> float:
        """
        Compute Backward Transfer (BWT).

        BWT measures how much learning new tasks affects performance on previous tasks.
        Negative BWT indicates forgetting (catastrophic forgetting).
        Positive BWT indicates forward knowledge transfer.

        Formula:
            BWT = (1/(t-1)) * Σ_{j=1}^{t-1} (R_{t,j} - R_{j,j})

        where:
        - R_{t,j} is accuracy on task j after learning task t
        - R_{j,j} is accuracy on task j right after learning it

        Interpretation:
        - BWT < 0: Forgetting (performance on old tasks degrades)
        - BWT ≈ 0: No forgetting, no transfer
        - BWT > 0: Positive backward transfer (rare, beneficial transfer)

        Returns:
            float: Backward transfer score (negative = forgetting)
        """
        if self.current_task < 1:
            return 0.0

        bwt_values = []

        for task_id in range(self.current_task):
            # Accuracy on task_id after learning current_task
            acc_after = self.accuracy_matrix[self.current_task, task_id]

            # Accuracy on task_id right after learning it (initial performance)
            acc_initial = self.accuracy_matrix[task_id, task_id]

            # Difference: positive = improvement, negative = forgetting
            bwt_values.append(acc_after - acc_initial)

        return np.mean(bwt_values)

    def compute_forward_transfer(self) -> float:
        """
        Compute Forward Transfer (FWT).

        FWT measures how knowledge from previous tasks helps learn new tasks.
        Requires baseline accuracy on each task before any training.

        For this implementation, we'll use a simplified version comparing
        initial performance on task t with final performance.

        Returns:
            float: Forward transfer score
        """
        # Simplified version - would need zero-shot baseline for full implementation
        fwt_values = []

        for task_id in range(1, self.current_task + 1):
            # Compare first time seeing task vs last time
            acc_first = self.accuracy_matrix[task_id, task_id]
            acc_last = self.accuracy_matrix[self.current_task, task_id]

            fwt_values.append(acc_last - acc_first)

        return np.mean(fwt_values) if fwt_values else 0.0

    def get_final_metrics(self) -> Dict:
        """
        Get final metrics after all tasks are learned.

        Returns:
            dict: Dictionary containing all final metrics
        """
        aa = self.compute_average_accuracy()
        bwt = self.compute_backward_transfer()

        # Compute per-task final accuracy
        final_accuracies = self.accuracy_matrix[self.num_tasks - 1, :]

        return {
            'average_accuracy': aa,
            'backward_transfer': bwt,
            'final_accuracies': final_accuracies,
            'accuracy_matrix': self.accuracy_matrix.copy(),
            'forgetting_measures': self._compute_forgetting_measures()
        }

    def _compute_forgetting_measures(self) -> np.ndarray:
        """
        Compute forgetting measure for each task.

        Forgetting for task j = max_{i<j} R_{i,j} - R_{t,j}
        where t is the final task

        Returns:
            np.ndarray: Forgetting measure for each task
        """
        forgetting = np.zeros(self.num_tasks)

        for task_id in range(self.num_tasks):
            # Best accuracy on this task during training
            try:
                best_acc = np.max(self.accuracy_matrix[:self.current_task, task_id])
            except ValueError:
                # If no historical tasks (empty array), use current accuracy as best
                best_acc = self.accuracy_matrix[self.current_task, task_id]

            # Final accuracy on this task
            final_acc = self.accuracy_matrix[self.num_tasks - 1, task_id]

            # Forgetting = drop from best to final
            forgetting[task_id] = best_acc - final_acc

        return forgetting

    def print_metrics_summary(self):
        """Print a formatted summary of all metrics."""
        print("\n" + "="*60)
        print("CONTINUAL LEARNING METRICS SUMMARY")
        print("="*60)

        final = self.get_final_metrics()

        print(f"\n[PERFORMANCE] Overall Performance:")
        print(f"   Average Accuracy (AA): {final['average_accuracy']:.4f} ({final['average_accuracy']*100:.2f}%)")
        print(f"   Backward Transfer (BWT): {final['backward_transfer']:.4f}")

        if final['backward_transfer'] < 0:
            print(f"   [WARNING] Catastrophic forgetting detected! (-{abs(final['backward_transfer'])*100:.2f}%)")
        else:
            print(f"   [OK] No significant forgetting")

        print(f"\n[TASKS] Per-Task Final Accuracies:")
        for task_id, acc in enumerate(final['final_accuracies']):
            print(f"   Task {task_id + 1}: {acc:.4f} ({acc*100:.2f}%)")

        print(f"\n[FORGETTING] Forgetting Measures:")
        for task_id, forget in enumerate(final['forgetting_measures']):
            if forget > 0:
                print(f"   Task {task_id + 1}: {forget:.4f} ({forget*100:.2f}% performance drop)")
            else:
                print(f"   Task {task_id + 1}: {forget:.4f} (no drop)")

        print("="*60 + "\n")

    def plot_accuracy_matrix(self, save_path=None):
        """
        Plot the accuracy matrix as a heatmap.

        Args:
            save_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(10, 8))

        # Only plot filled portion of matrix
        max_task = self.current_task
        matrix_to_plot = self.accuracy_matrix[:max_task, :max_task]

        sns.heatmap(matrix_to_plot, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'Task {i+1}' for i in range(max_task)],
                    yticklabels=[f'After T{i+1}' for i in range(max_task)],
                    vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})

        plt.xlabel('Test Task', fontsize=12)
        plt.ylabel('Training Progress', fontsize=12)
        plt.title('Continual Learning Accuracy Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Accuracy matrix plot saved to {save_path}")
        else:
            print("[INFO] No save path provided, skipping plot save")

        plt.close()

    def plot_learning_curve(self, save_path=None):
        """
        Plot learning curves showing accuracy progression.

        Args:
            save_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(12, 6))

        max_task = self.current_task
        colors = plt.cm.tab10(np.linspace(0, 1, max_task))

        for task_id in range(max_task):
            # Get accuracy progression for this task
            accuracies = []
            for training_step in range(task_id, max_task + 1):
                accuracies.append(self.accuracy_matrix[training_step, task_id])

            plt.plot(range(task_id, max_task + 1), accuracies,
                    marker='o', label=f'Task {task_id + 1}', color=colors[task_id], linewidth=2)

        plt.xlabel('Task Learned', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Continual Learning Performance Progression', fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Learning curve plot saved to {save_path}")
        else:
            print("[INFO] No save path provided, skipping plot save")

        plt.close()


def compute_average_accuracy(model, task_loaders: List[Dict], device='cpu') -> float:
    """
    Convenience function to compute average accuracy across all tasks.

    Args:
        model: Trained model
        task_loaders: List of task dictionaries with test loaders
        device: Device to run evaluation on

    Returns:
        float: Average accuracy across all tasks
    """
    num_tasks = len(task_loaders)
    metrics = ContinualLearningMetrics(num_tasks)

    # Set current task to last task for final evaluation
    metrics.current_task = num_tasks - 1

    # Evaluate on all tasks
    for task_id, task_info in enumerate(task_loaders):
        test_loader = task_info['test_loader']
        accuracy = metrics.evaluate_task(model, task_id, test_loader, device)
        metrics.accuracy_matrix[metrics.current_task, task_id] = accuracy

    return metrics.compute_average_accuracy()


def compute_backward_transfer(accuracy_matrix: np.ndarray) -> float:
    """
    Compute BWT from a pre-computed accuracy matrix.

    Args:
        accuracy_matrix (np.ndarray): Matrix where element [i,j] is accuracy on task j after learning task i

    Returns:
        float: Backward transfer score
    """
    num_tasks = accuracy_matrix.shape[0]

    if num_tasks < 2:
        return 0.0

    bwt_values = []

    for task_id in range(num_tasks - 1):
        acc_final = accuracy_matrix[num_tasks - 1, task_id]
        acc_initial = accuracy_matrix[task_id, task_id]
        bwt_values.append(acc_final - acc_initial)

    return np.mean(bwt_values)


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Continual Learning Metrics - Testing")
    print("="*60)

    # Simulate training on 3 tasks
    num_tasks = 3
    metrics = ContinualLearningMetrics(num_tasks)

    # Simulated accuracy matrix (typical continual learning scenario with forgetting)
    # Rows: after learning task 0, 1, 2
    # Cols: accuracy on task 0, 1, 2
    simulated_matrix = np.array([
        [0.95, 0.00, 0.00],  # After task 0: only task 0 learned
        [0.85, 0.92, 0.00],  # After task 1: task 0 forgot 10%, task 2 not learned
        [0.70, 0.88, 0.90]   # After task 2: task 0 forgot 25%, task 1 forgot 4%
    ])

    metrics.accuracy_matrix = simulated_matrix
    metrics.current_task = num_tasks

    # Compute and display metrics
    metrics.print_metrics_summary()

    # Test individual metric computations
    print("\n[METRICS] Individual Metrics:")
    print(f"   Average Accuracy: {metrics.compute_average_accuracy():.4f}")
    print(f"   Backward Transfer: {metrics.compute_backward_transfer():.4f}")

    print("\n" + "="*60)
    print("Metrics test completed successfully!")
    print("="*60)
