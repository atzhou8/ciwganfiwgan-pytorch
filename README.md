# ciwganfiwgan-pytorch

This is a PyTorch implementation of **Categorical Info WaveGAN (ciwGAN)** and **Featural Info WaveGAN (fiwGAN)** from [Beguš, 2021](https://www.sciencedirect.com/science/article/pii/S0893608021001052). The original code (in Tensorflow 1.12) can be found [here](https://github.com/gbegus/fiwGAN-ciwGAN).

## Usage

### Training WaveGAN
```
python train.py --datadir training_directory --logdir log_directory
```

### Training ciwGAN
```
python train.py --ciw --num_categ N --datadir training_directory --logdir log_directory
```

### Training fiwGAN
```
python train.py --fiw --num_categ N --datadir training_directory --logdir log_directory
```

### Continue training from a previous checkpoint

Add `--cont last` to the end of the training statement to continue from the last found state  
Add `--cont epoch_number` to continue from the state corresponding to `epoch_number`
