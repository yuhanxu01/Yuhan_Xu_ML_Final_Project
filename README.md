# Continual Learning: Mitigating Catastrophic Forgetting

A comparative study of three strategies to address catastrophic forgetting in neural networks using the UCI HAR dataset.

## Overview

This project implements and compares three continual learning approaches:
- **Naive Fine-tuning**: Baseline sequential learning
- **Frozen Backbone**: Architecture-based approach 
- **Experience Replay**: Memory buffer with 100 samples per task

## Results

| Strategy | Avg Accuracy | Backward Transfer | Task 1 | Task 2 | Task 3 |
|----------|--------------|-------------------|--------|--------|--------|
| Naive | 0.724 | -0.189 | 0.856 | 0.688 | 0.628 |
| Frozen | 0.783 | -0.112 | 0.892 | 0.756 | 0.701 |
| Replay | 0.847 | -0.043 | 0.878 | 0.835 | 0.828 |

**Key Findings:**
- Experience Replay improves accuracy by 12.3% over baseline
- Reduces catastrophic forgetting by 77.2%
- Shows best balance between learning new tasks and retaining old knowledge

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Setup (Auto-download dataset)
```bash
# Windows
bash setup.sh

# Linux/Mac  
chmod +x setup.sh && ./setup.sh
```

### Run Experiments
```bash
python main.py
```

## Project Structure
```
.
├── src/
│   ├── dataset.py      # Data loading
│   ├── model.py        # Neural network
│   ├── metrics.py      # Evaluation (AA, BWT)
│   └── strategies.py   # 3 learning strategies
├── main.py             # Main script
├── setup.sh            # Auto-download script
└── requirements.txt    # Dependencies
```

## Dataset

UCI Human Activity Recognition (HAR) Dataset
- 6 activity classes split into 3 sequential tasks
- 561-dimensional features
- 7,352 training samples, 2,947 test samples

**Auto-download**: The first time you run the code, the dataset downloads automatically.

## Neural Network

```
Input(561) → Linear(256) → ReLU → Dropout(0.2) → 
Linear(256) → ReLU → Dropout(0.2) → Linear(6) → Output
```

- Parameters: 211,206
- Training: Adam, lr=0.001, 10 epochs per task

## Evaluation Metrics

**Average Accuracy (AA)**: Overall performance across all tasks
**Backward Transfer (BWT)**: Forgetting measure (negative = forgetting)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, scikit-learn, requests, tqdm, matplotlib, seaborn

## Author

**Yuhan Xu**  
Fordham University  
CISC 5800 - Machine Learning  
2026

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{continual_learning_har,
  title={Mitigating Catastrophic Forgetting in Continual Learning},
  author={Xu, Yuhan},
  year={2026},
  url={https://github.com/yourusername/continual-learning-har}
}
```

---

**Keywords**: continual-learning, catastrophic-forgetting, pytorch, machine-learning
