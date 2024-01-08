
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
    parser.add_argument("--n_samples", type=int, default=1000, 
                        help="inb of samples to take in initial dataset (too heavy otherwise)")
    parser.add_argument("--model", type=str, choices=['distilbert','bert','albert'],default='distilbert')
    parser.add_argument("--id_p", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--id_v", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--n_models", type=int, default=None, 
                        help="if PET/iPET, number of models to train per pvp")
    parser.add_argument("--lr", type=float, default=1e-5, 
                        help="learning rate")
    parser.add_argument("--bsize_train", type=int, default=8, 
                        help="chosen batch size for examples training")
    parser.add_argument("--bsize_test", type=int, default=32, 
                        help="chosen batch size for testing")
    parser.add_argument("--train",action=argparse.BooleanOptionalAction ,default=True )
    parser.add_argument("--test",action=argparse.BooleanOptionalAction ,default=True )
    parser.add_argument("--val",action=argparse.BooleanOptionalAction ,default=False,\
                        help="to activate or not validation during training (32//bsize epochs) and gain time" )
    parser.add_argument("--save",action=argparse.BooleanOptionalAction ,default=False )

    args = parser.parse_args()

    n_samples=args.n_samples
    raw_data = load_dataset("scikit-learn/imdb", split="train")
    raw_data=raw_data.shuffle()
    model_prefix=args.model
    if model_prefix =='distilbert': 
       model_name="distilbert-base-uncased"
       tokenizer=DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_prefix =='bert': 
       model_name="bert-base-uncased"
       tokenizer=BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_prefix=='albert':
       model_name="albert-base-v2"
       tokenizer=AlbertTokenizer.from_pretrained(model_name, do_lower_case=True) 


    if (args.id_p is not None) and (args.id_v is not None) : 
       print('\n'+'-'*25 + 'Pattern Exploiting mode'+'-'*25 )
       print(f'id pattern = {args.id_p}, id verbalizer = {args.id_v}')
       pvp=PVP(id_p=args.id_p,id_v=args.id_v)
       if model_prefix=='distilbert':mlm = DistilBertForMaskedLM.from_pretrained(model_name)
       if model_prefix=='bert': mlm= BertForMaskedLM.from_pretrained(model_name)
       if model_prefix=='albert':mlm = AlbertForMaskedLM.from_pretrained(model_name)
       model=PETClassifier(mlm,tokenizer,pvp)
    else :    
       print('\n'+'-'*25 + 'Classical mode'+'-'*25 )  
       pvp=None
       if model_prefix=='distilbert':bert = DistilBertModel.from_pretrained(model_name)
       if model_prefix=='bert':bert = BertModel.from_pretrained(model_name)
       if model_prefix=='albert':bert = AlbertModel.from_pretrained(model_name)
       model=Classifier(bert)


    # prepare data
    dataset=CustomDataset(tokenizer,raw_data,pvp=pvp,n_samples=n_samples) # normal dataset
    # train/test split
    examples, test = Subset(dataset, range(32)),Subset(dataset, range(32, len(dataset)))
    # set up loaders
    data_collator = DataCollator(dataset.tokenizer)
    bsize_train=args.bsize_train
    bsize_test=args.bsize_test
    examples_loader = DataLoader(examples, batch_size=bsize_train, collate_fn=data_collator) # train for 1 batch
    test_loader = DataLoader(test, batch_size=bsize_test, collate_fn=data_collator)

    

    if args.train:
       train(model, examples_loader, test_loader,bsize=bsize_train, lr=args.lr,save=args.save) 

    
  
    
    
    
 
    


