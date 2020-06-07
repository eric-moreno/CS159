# Intro

PyTorch implementation of [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474).

## Run

Determine the parameters needed for the meta-learner. For example: 

  - Specify output directory with --outdir
  - Specify RNN type with --RNN
  - Specify layers with --num_layers
  - Specify cuda with --no-cuda
  - Many more options available in the main.py code! 
```bash
python main.py --outdir training_directory --RNN GRU --num_layers 2
```

