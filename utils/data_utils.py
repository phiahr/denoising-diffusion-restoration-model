import torch
import torch.nn.functional as F
import platform
import numpy as np
import matplotlib.pyplot as plt
from ignite.engine import *
from torch.utils.data import Dataset
import os


def eval_step(engine, batch):
    return batch

def uniform_row_conv1d_matrix(input_shape, kernel_len, padding):
    conv_matrix = torch.zeros(size = (input_shape[1], input_shape[1]-2*padding))
    multiplier = 1/(kernel_len)
    for col in range(0, conv_matrix.shape[1]):
        start = int(np.max([col-kernel_len//2+1,0]))
        end = int(np.min([col+kernel_len//2+1, conv_matrix.shape[0]]))
        conv_matrix[start:end, col] = multiplier
    return conv_matrix


def set_device(device=None):
    # define devices
    if device is None:
      if platform.system() == 'Darwin' and torch.backends.mps.is_available():
          # use hardware acceleration for MacOS
          device = 'mps:0'
      else:
          # use hardware acceleration with cuda drivers (i.e. Linux and Windows under NVIDIA GPUs)
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      if device == 'cpu':
        device = 'cpu'
      elif device == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
      else:
        device = 'cpu'
        print(f"Unknown device {device}. Using CPU instead.")
      device = device

    print(f"Using device: {device}")
    return device


def plot_batches(batches: list):
  '''
  :params 
    batchtes: list of batches to plot

  :returns None
  '''

  fig, axes = plt.subplots(len(batches), batches[0].shape[0])
  
  for i, batch in enumerate(batches):

    # Iterate over the pictures
    for j in range(batch.shape[0]):
        
        # Extract the i-th image tensor
        image_tensor = batch[j]
        
        # Convert PyTorch tensor to NumPy array
        image_numpy = image_tensor.permute(1, 2, 0).cpu().detach().numpy()
        
        axes[i, j].imshow(image_numpy)
        axes[i, j].axis('off')  # Turn off axis labels
        
  plt.show()