
import argparse
import os
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
import random

seed_value = 123
random.seed(seed_value)
# torch.manual_seed(seed_value)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--n_samples", type=int, default=1000, 
                        help="inb of samples to take in initial dataset (too heavy otherwise)")
    parser.add_argument("--dataset", type=str,default='imdb', choices=['imdb', 'boolq'])
    parser.add_argument("--model", type=str, choices=['distilbert','bert','albert'],default='distilbert')
    parser.add_argument("--id_p", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--id_v", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--num_model", type=int, default=None, 
                        help="if PET/iPET, number identifier of the model to test")
    parser.add_argument("--K", type=int, default=None, 
                        help="if PET/iPET, number of independant models to aggregate results from ")
    parser.add_argument("--bsize_test", type=int, default=32, 
                        help="chosen batch size for testing")

    args = parser.parse_args()


    n_samples=args.n_samples
    ds_name=args.dataset
    raw_data = load_raw_data(ds_name)
    raw_data=raw_data.shuffle(seed=seed_value)
    model_name,tokenizer=create_name_tokenizer(args)

    # prepare data
    if (args.id_p is not None) and (args.id_v is not None) :
       print('\n'+'-'*25 + 'Pattern Exploiting test'+'-'*25 ) 
       print(f'id pattern = {args.id_p}, id verbalizer = {args.id_v}')
       pvp=PVP(args.id_p,args.id_v, dataset=ds_name)
    else:
       print('\n'+'-'*25 + 'Classical test'+'-'*25 )
       pvp=None
    dataset=CustomDataset(tokenizer,raw_data,pvp=pvp,n_samples=n_samples, name=ds_name) # normal dataset
   #  print(dataset.observe(0))
    # train/test split
    _, test_set = Subset(dataset, range(32)),Subset(dataset, range(32, len(dataset)))
    # set up loaders
    data_collator = DataCollator(dataset.tokenizer)
    bsize_test=args.bsize_test
    test_loader = DataLoader(test_set, batch_size=bsize_test, collate_fn=data_collator)

    if args.K is None:

      save_path=get_save_path(args,args.num_model)

      backbone=create_backbone(args,model_name)
      if (args.id_p is not None) and (args.id_v is not None):model=PETClassifier(backbone,tokenizer,pvp)
      else:model=Classifier(backbone)
      model.load_state_dict(torch.load(save_path))
      model=model.to(DEVICE)


      l,a=test(model,test_loader)
      print("Single test")
      print( "test loss: {:.4f}".format(l),"test acc:{:.4f}".format(a * 100))

    else:
       
       list_models=[]
       for k in range(args.K):
          save_path=get_save_path(args,k+1)
          backbone=create_backbone(args,model_name,tokenizer,pvp)
          if (args.id_p is not None) and (args.id_v is not None):model=PETClassifier(backbone,tokenizer,pvp)
          else:model=Classifier(backbone)
          model.load_state_dict(torch.load(save_path))
          model=model.to(DEVICE)
          list_models.append(model)

       l,a=test_ensemble(list_models,test_loader)
       print("Ensemble test")
       print( "test loss: {:.4f}".format(l),"test acc:{:.4f}".format(a * 100))
       
      

    
       
       

    
  
    
    
    
 
    


