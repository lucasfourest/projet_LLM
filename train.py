
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
    parser.add_argument("--dataset", type=str,default='imdb', choices=['imdb', 'boolq'])
   #  parser.add_argument("--few_shot",action=argparse.BooleanOptionalAction ,default=False,\
   #                      help="to perform few shot training" )
   #  parser.add_argument("--normal",action=argparse.BooleanOptionalAction ,default=False,\
   #                      help="to perform normal training" )
    parser.add_argument("--n_samples", type=int, default=5000, 
                        help="nb of samples to take in initial dataset (too heavy otherwise)")
    parser.add_argument("--n_examples", type=int, default=32, 
                        help="nb of training examples fro few shot training")
    parser.add_argument("--model", type=str, choices=['distilbert','bert','albert'],default='distilbert')
    parser.add_argument("--id_p", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--id_v", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--num_model", type=int, default=None, 
                        help="if PET/iPET, number identifier of the model to train")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=1, 
                        help="number of epochs, that is number of complete passes over all examples")
    parser.add_argument("--bsize_train", type=int, default=8, 
                        help="chosen batch size for examples training")
    parser.add_argument("--bsize_test", type=int, default=32, 
                        help="chosen batch size for testing")
    parser.add_argument("--eval",action=argparse.BooleanOptionalAction ,default=False,\
                        help="to activate or not evaluation at end of training and gain time" )
    parser.add_argument("--eval_every",type=int ,default=10,\
                        help="for normal training" )
    parser.add_argument("--save",action=argparse.BooleanOptionalAction ,default=False )

    args = parser.parse_args()

    few_shot=(args.id_p is not None) and (args.id_v is not None)
    normal = not few_shot
    n_samples=args.n_samples
    ds_name=args.dataset
    raw_data = load_raw_data(ds_name)
    raw_data=raw_data.shuffle(seed=seed_value)
    
    model_name,tokenizer=create_name_tokenizer(args)
    save_path=get_save_path(args,args.num_model)
    backbone=create_backbone(args,model_name)

    if (args.id_p is not None) and (args.id_v is not None) : 
       print('\n'+'-'*25 + 'Pattern Exploiting train'+'-'*25 )
       print(f'id pattern = {args.id_p}, id verbalizer = {args.id_v}')
       pvp=PVP(id_p=args.id_p,id_v=args.id_v,dataset=ds_name)
       mlm=backbone
       model=PETClassifier(mlm,tokenizer,pvp)
    else :    
       print('\n'+'-'*25 + 'Classical train'+'-'*25 )  
       pvp=None
       bert=backbone
       model=Classifier(bert)

    

    if normal:
      # prepare data
      dataset=CustomDataset(tokenizer,raw_data,pvp=pvp,n_samples=n_samples, name=ds_name) # normal dataset
      # train/test split
      examples, test = Subset(dataset, range(args.n_examples)),Subset(dataset, range(args.n_examples, len(dataset)))
      # set up loaders
      data_collator = DataCollator(dataset.tokenizer)
      bsize_train=args.bsize_train
      bsize_test=args.bsize_test
      examples_loader = DataLoader(examples, batch_size=bsize_train, collate_fn=data_collator) # train for 1 batch
      test_loader = DataLoader(test, batch_size=bsize_test, collate_fn=data_collator)

      

      train(model, examples_loader, test_loader,n_epochs=args.n_epochs, lr=args.lr,eval_every=args.eval_every,save_path=save_path)  
   

    if few_shot:
       # prepare data
      dataset=CustomDataset(tokenizer,raw_data,pvp=pvp,n_samples=n_samples, name=ds_name) # normal dataset
      # train/test split
      examples, test = Subset(dataset, range(args.n_examples)),Subset(dataset, range(args.n_examples, len(dataset)))
      # set up loaders
      data_collator = DataCollator(dataset.tokenizer)
      bsize_train=args.bsize_train
      bsize_test=args.bsize_test
      examples_loader = DataLoader(examples, batch_size=bsize_train, collate_fn=data_collator) # train for 1 batch
      test_loader = DataLoader(test, batch_size=bsize_test, collate_fn=data_collator)

      

      
      train(model, examples_loader, test_loader,n_epochs=args.n_epochs, lr=args.lr,eval_every=args.eval_every,save_path=save_path)  

  
   
    
    
    
 
    


