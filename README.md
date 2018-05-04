# Setup

1. Install Anaconda 3
2. Clone this repository
3. `conda env create`

# Usage

1. Generate dataset: `python generate_dataset.py`
2. Train model: `python train.py`
3. Start TensorBoard: `tensorboard --logdir=data/tensorboard_logs`
4. Evaluate model: `python evaluate.py`
5. Inference: `python predict.py`
