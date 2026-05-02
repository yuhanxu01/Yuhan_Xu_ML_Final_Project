"""
Main Training Script for Continual Learning Experiments
Compares three strategies: Naive, Frozen Backbone, Experience Replay
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.dataset import get_dataloaders
from src.model import MLP
from src.strategies import create_strategy
from src.metrics import ContinualLearningMetrics


class ContinualLearningExperiment:
    """Run continual learning experiments and evaluate strategies."""

    def __init__(self, device='cpu', learning_rate=0.001, epochs=10,
                 batch_size=64, buffer_size=100, random_seed=42):
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.random_seed = random_seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        self.results = {}
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Experiment initialized")
        print(f"Device: {self.device}")
        print(f"Epochs per task: {self.epochs}")
        print(f"Buffer size: {self.buffer_size}")

    def load_data(self):
        """Load HAR dataset and create task dataloaders."""
        print("\nLoading UCI HAR Dataset...")
        self.tasks = get_dataloaders(
            batch_size=self.batch_size,
            normalize=True,
            random_seed=self.random_seed
        )

        print(f"Loaded {len(self.tasks)} tasks")
        for task_info in self.tasks:
            task_id = task_info['task_id']
            classes = task_info['classes']
            print(f"  Task {task_id + 1}: Classes {classes}")

        return self.tasks

    def run_strategy(self, strategy_name):
        """Train and evaluate one strategy."""
        print(f"\n{'='*60}")
        print(f"Running Strategy: {strategy_name.upper()}")
        print(f"{'='*60}")

        model = MLP(input_dim=561, hidden_dim=256, output_dim=6)

        if strategy_name == 'replay':
            strategy = create_strategy(
                strategy_name, model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs,
                buffer_size=self.buffer_size
            )
        else:
            strategy = create_strategy(
                strategy_name, model, device=self.device,
                learning_rate=self.learning_rate, epochs=self.epochs
            )

        metrics = ContinualLearningMetrics(num_tasks=len(self.tasks))

        for task_info in self.tasks:
            task_id = task_info['task_id']
            train_loader = task_info['train_loader']
            test_loader = task_info['test_loader']

            strategy.train_task(train_loader, test_loader, task_id)

            for eval_task_id in range(task_id + 1):
                eval_loader = self.tasks[eval_task_id]['test_loader']
                accuracy = strategy.evaluate(eval_loader, eval_task_id)
                metrics.accuracy_matrix[task_id, eval_task_id] = accuracy

        final_metrics = metrics.get_final_metrics()

        result = {
            'strategy_name': strategy_name,
            'accuracy_matrix': metrics.accuracy_matrix.copy(),
            'average_accuracy': final_metrics['average_accuracy'],
            'backward_transfer': final_metrics['backward_transfer'],
            'final_accuracies': final_metrics['final_accuracies'],
            'forgetting_measures': final_metrics['forgetting_measures']
        }

        self.results[strategy_name] = result

        print(f"\nResults:")
        print(f"  Average Accuracy: {result['average_accuracy']:.4f}")
        print(f"  Backward Transfer: {result['backward_transfer']:.4f}")

        return result

    def run_all_strategies(self, strategies=['naive', 'frozen', 'replay']):
        """Run all strategies."""
        print(f"\nRunning {len(strategies)} strategies...")

        for strategy_name in strategies:
            try:
                self.run_strategy(strategy_name)
            except Exception as e:
                print(f"Strategy {strategy_name} failed: {e}")

        return self.results

    def plot_results(self):
        """Generate comparison plots."""
        print("\nGenerating comparison plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continual Learning Strategies Comparison',
                    fontsize=16, fontweight='bold')

        strategies = list(self.results.keys())
        aa_scores = [self.results[s]['average_accuracy'] for s in strategies]
        bwt_scores = [self.results[s]['backward_transfer'] for s in strategies]
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

        # Plot 1: Average Accuracy
        ax1 = axes[0, 0]
        bars = ax1.bar(strategies, aa_scores, color=colors[:len(strategies)], alpha=0.7)
        ax1.set_ylabel('Average Accuracy', fontsize=12)
        ax1.set_title('Average Accuracy Comparison', fontsize=13, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)

        for bar, score in zip(bars, aa_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 2: Backward Transfer
        ax2 = axes[0, 1]
        bars = ax2.bar(strategies, bwt_scores, color=colors[:len(strategies)], alpha=0.7)
        ax2.set_ylabel('Backward Transfer', fontsize=12)
        ax2.set_title('Backward Transfer Comparison', fontsize=13, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.grid(True, alpha=0.3)

        for bar, score in zip(bars, bwt_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center',
                    va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')

        # Plot 3: Per-task accuracies
        ax3 = axes[1, 0]
        task_ids = np.arange(len(self.tasks))
        width = 0.25

        for i, strategy in enumerate(strategies):
            final_accs = self.results[strategy]['final_accuracies']
            offset = (i - len(strategies)/2 + 0.5) * width
            ax3.bar(task_ids + offset, final_accs, width,
                   label=strategy.capitalize(), color=colors[i], alpha=0.7)

        ax3.set_xlabel('Task', fontsize=12)
        ax3.set_ylabel('Final Accuracy', fontsize=12)
        ax3.set_title('Per-Task Final Accuracies', fontsize=13, fontweight='bold')
        ax3.set_xticks(task_ids)
        ax3.set_xticklabels([f'Task {i+1}' for i in task_ids])
        ax3.legend(loc='lower right')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)

        # Plot 4: Learning curves
        ax4 = axes[1, 1]
        for i, strategy in enumerate(strategies):
            acc_matrix = self.results[strategy]['accuracy_matrix']
            diagonal = np.diag(acc_matrix)
            ax4.plot(task_ids, diagonal, marker='o',
                    label=strategy.capitalize(), color=colors[i],
                    linewidth=2, markersize=8)

        ax4.set_xlabel('Task', fontsize=12)
        ax4.set_ylabel('Accuracy at Task Completion', fontsize=12)
        ax4.set_title('Learning Progression', fontsize=13, fontweight='bold')
        ax4.set_xticks(task_ids)
        ax4.set_xticklabels([f'Task {i+1}' for i in task_ids])
        ax4.legend(loc='lower right')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")

    def print_summary(self):
        """Print final summary."""
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)

        sorted_strategies = sorted(
            self.results.keys(),
            key=lambda x: self.results[x]['average_accuracy'],
            reverse=True
        )

        print("\nStrategies ranked by Average Accuracy:")
        for rank, strategy in enumerate(sorted_strategies, 1):
            result = self.results[strategy]
            print(f"  {rank}. {strategy.upper()}: AA={result['average_accuracy']:.4f}, BWT={result['backward_transfer']:.4f}")

        print("\nPer-Task Final Accuracies:")
        for strategy in sorted_strategies:
            result = self.results[strategy]
            print(f"  {strategy.upper()}: ", end="")
            for i, acc in enumerate(result['final_accuracies']):
                print(f"T{i+1}={acc:.3f}", end=" " if i < len(result['final_accuracies'])-1 else "\n")

        print("="*60)


def main():
    """Main execution function."""
    print("="*60)
    print("CONTINUAL LEARNING EXPERIMENTS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 64,
        'buffer_size': 100,
        'random_seed': 42
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    experiment = ContinualLearningExperiment(**config)
    experiment.load_data()
    experiment.run_all_strategies(['naive', 'frozen', 'replay'])
    experiment.plot_results()
    experiment.print_summary()

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
