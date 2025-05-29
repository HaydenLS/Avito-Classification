# Main imports
import os
import numpy as np
import pandas as pd

# Label encoder
import pickle
from sklearn.preprocessing import LabelEncoder

# Torch
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score

# Images and graphics
from matplotlib import colors, pyplot as plt

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

DEVICE = torch.device("cuda")