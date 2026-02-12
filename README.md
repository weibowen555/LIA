# LIA: A Logical-Rule Autoencoder for Interpretable Recommendations

This repository contains the implementation of **LIA (Logical-rule Interpretable Autoencoder)**, a novel recommendation model that bridges the gap between neural recommendation accuracy and rule-based interpretability. LIA learns a set of logical AND/OR rules from implicit user-item interaction data through a learnable logical rule layer with differentiable gates, enabling end-to-end training while producing discrete, human-readable rules at inference time.

## Key Contributions

- **Learnable Logical Rule Layer**: Each neuron learns whether to perform conjunction (AND) or disjunction (OR) via a differentiable sigmoid gate, producing discrete interpretable rules at inference while maintaining gradient flow during training.
- **Signed-Weight Negation**: Negation is encoded in the sign of connection weights (W in [-1, 1]), enabling NOT operations without doubling the input dimension.
- **Disjunctive Aggregation**: A linear aggregation layer combines rule outputs to reconstruct user preferences, supporting both conjunctive and disjunctive reasoning over item interactions.


## Datasets

| Dataset | Users | Items | Density |
|---------|-------|-------|---------|
| MovieLens 100K (ML100K) | 943 | 1,682 | 6.30% |
| MovieLens 1M (ML1M) | 6,040 | 3,706 | 4.47% |
| Yelp | 12,171 | 9,252 | 0.38% |

Datasets should be placed in `data/recsys_data/`. Evaluation uses a leave-k-out split strategy (default k=5) with popularity-ordered sampling.

## Quick Start

### Requirements

- Python 3.10+
- PyTorch
- NumPy, SciPy

### Training

**Single GPU:**
```bash
python main_1.py
```

**With SLURM:**
```bash
sbatch run.slurm
```

### Configuration

Training is configured via two config files:

- `main_config.cfg` — dataset, epochs, evaluation metrics, early stopping
- `model_config/LIA.cfg` — model hyperparameters

Key hyperparameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `structure` | Rule layer sizes (e.g., `0@1000`) | `0@1000` |
| `lr` | Learning rate | `0.0001` |
| `use_not` | Enable signed-weight negation | `True` |
| `use_learnable_gate` | Learnable AND/OR gate per neuron | `True` |



