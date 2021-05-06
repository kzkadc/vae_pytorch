# Variational Autoencoder
PyTorch implementation of Variational Autoencoder (VAE).

## Requirements
- PyTorch
- PyTorch-Ignite
- Matplotlib
- NumPy

## Usage
1. Run `main.py` to train VAE model.
  - You can choose the fully-connected model or the CNN model.
  - Pretrained sample models are included in this repository.
  
```bash
$ python .\main.py -h
usage: main.py [-h] [-e E] [-b B] [--zdim ZDIM] -m {fc,cnn} [-o O]

optional arguments:
  -h, --help   show this help message and exit
  -e E         epoch
  -b B         batch size
  --zdim ZDIM  number of dimensions of latent space
  -m {fc,cnn}  model architecture
  -o O         ouput directory
  
# example
$ python main.py -e 50 -b 64 --zdim 20 -m fc -o fc_result
$ python main.py -e 50 -b 64 --zdim 20 -m cnn -o cnn_result
```

2. Generate images with trained models on `generate_images.ipynb` notebook.
