# Continual Learning
This project aims to investigate how normalisation and continual backpropergation affect the plasticity of a neural network. 

## Installation
### Baselines
 * Weight decay (sgd/adam) 
 * Continual backpropergation
 * Layer norm

### Experiments
 * Slippery Ant v5
 * Sine regression
 * Continual time-delays

### Notes
 * Clone repository: `git clone https://github.com/LucMc/continual-learning.git`
 * Install with: `pip install -e .`
 * The todo list is currently at the top of `optim/continual_backprop.py`

### Example Usage
```
python sine_exp.py --no-debug --methods "adam" "cbp"`
```
