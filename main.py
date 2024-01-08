
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, Subset
from datasets import load_dataset
from tqdm.autonotebook import *
from transformers import AlbertForMaskedLM, AlbertTokenizer, BertForMaskedLM, BertTokenizer, \
  DistilBertForMaskedLM, DistilBertTokenizer

from pvp import *
from dataset import *
from utils import *
from model import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--id_p", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--id_v", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--n_models", type=int, default=None, 
                        help="if PET/iPET, number of models to train per pvp")
    parser.add_argument("--bsize_train", type=int, default=8, 
                        help="chosen batch size for examples training")
    parser.add_argument("--bsize_test", type=int, default=32, 
                        help="chosen batch size for testing")

    args = parser.parse_args()
    


