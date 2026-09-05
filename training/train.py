"""
Training Loop Module
Trains the HybridSignModel on extracted landmarks.

Features:
  - Stratified train/validation split
  - Label smoothing
  - AdamW optimizer with weight decay
  - Learning rate scheduling (CosineAnnealingWarmRestarts)
  - Early stopping
  - Model checkpoint saving (best + periodic)
  - Gradient clipping
  - Mixed precision training (fp16)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import os
import time

from training.dataset import ISLLandmarkDataset, load_dataset
from training.model import HybridSignModel
