
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
    parser.add_argument("--n_examples", type=int, default=32, help="nb of examples")
    parser.add_argument("--dataset", type=str,default='imdb', choices=['imdb', 'boolq'])
    parser.add_argument("--model", type=str, choices=['distilbert','bert','albert'],default='distilbert')
    parser.add_argument("--id_p", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument("--id_v", type=int, default=None, 
                        help="id of pattern")
    parser.add_argument('--id_p_list', nargs='+', type=int, default=None,help='list of pattern idx \
                        (for mlm ensembling across several pvps)')
    parser.add_argument('--id_v_list', nargs='+', type=int, default=None, help='list of verbalizer idx \
                        (for mlm ensembling across several pvps)')
    parser.add_argument("--bsize", type=int, default=32, 
                        help="chosen batch size for testing")

    args = parser.parse_args()

    pet_ensemble=(args.id_p_list is not None) and (args.id_v_list is not None)
    pet_or_normal = not pet_ensemble
    n_samples=args.n_samples
    ds_name=args.dataset
    raw_data = load_raw_data(ds_name)
    raw_data=raw_data.shuffle(seed=seed_value)
    model_name,tokenizer=create_name_tokenizer(args)

    

    if pet_or_normal:
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
      example_set, test_set = Subset(dataset, range(args.n_examples)),Subset(dataset, range(args.n_examples, len(dataset)))
      # set up loaders
      data_collator = DataCollator(dataset.tokenizer)
      bsize=args.bsize
      test_loader = DataLoader(test_set, batch_size=bsize, collate_fn=data_collator)
      save_path=get_save_path(args)

      backbone=create_backbone(args,model_name ,id_p=args.id_p, id_v=args.id_v)
      if (args.id_p is not None) and (args.id_v is not None):model=PETClassifier(backbone,tokenizer,pvp)
      else:model=Classifier(backbone)
      model.load_state_dict(torch.load(save_path))
      model=model.to(DEVICE)


      l,a=test(model,test_loader)
      print("Single test")
      print( "test loss: {:.4f}".format(l),"test acc:{:.4f}".format(a * 100))



    if pet_ensemble:
       list_p=args.id_p_list
       list_v=args.id_v_list
       W_acc=[]
       models=[]
       for p in list_p:
          for v in list_v:
            save_path='models/'+args.model+'/'+args.dataset+'/'       
            save_path=save_path+'pet_pvp='+str(p)+str(v)+'/'
            save_path=save_path+'model.pth'
   
            pvp=PVP(p,v, dataset=ds_name)
            backbone=create_backbone(args,model_name,id_p=p, id_v=v)
            model=PETClassifier(backbone,tokenizer,pvp)
            model.load_state_dict(torch.load(save_path))
            model=model.to(DEVICE)
            models.append(model)

            dataset=CustomDataset(tokenizer,raw_data,pvp=pvp,n_samples=n_samples, name=ds_name) # normal dataset
            #  print(dataset.observe(0))
            # train/test split
            example_set, test_set = Subset(dataset, range(args.n_examples)),Subset(dataset, range(args.n_examples, len(dataset)))
            data_collator = DataCollator(dataset.tokenizer)
            bsize=args.bsize
            test_loader = DataLoader(test_set, batch_size=bsize, collate_fn=data_collator)
            example_loader = DataLoader(example_set, batch_size=bsize, collate_fn=data_collator)


            silly_model=PETClassifier(create_backbone(args,model_name,id_p=p, id_v=v),tokenizer,pvp)
            silly_model=silly_model.to(DEVICE)
            _,w_acc=test(silly_model,example_loader)
            W_acc.append(w_acc)
            # accuracy ponderation weights (cf PET global pred accros multiple pattern) are taken on 
            # known examples BEFORE training

       l,a=test_ensemble(models,W_acc,test_loader)
       print(l,a)

   
    
       
       

    
  
    
    
    
 
    


