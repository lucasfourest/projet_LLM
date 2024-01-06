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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Classifier(nn.Module):
    def __init__(self, mlm,tokenizer,pvp):
      super(Classifier, self).__init__()
      self.mlm = mlm # "masked language model"
      self.tokenizer = tokenizer
      self.id_pos, self.id_neg=self.tokenizer.encode(pvp.verbalizer.answer_pos,add_special_tokens=False,),\
                                self.tokenizer.encode(pvp.verbalizer.answer_neg,add_special_tokens=False,)
      # self.d_embed=self.mlm.config.hidden_size
      # self.head=nn.Sequential(
      #     nn.Dropout(0.3),
      #     nn.Linear(self.d_embed,self.tokenizer.vocab_size)
      # )

    def forward(self, input_ids, attention_mask, **kwargs):
      output_logits=self.mlm(input_ids=input_ids,attention_mask=attention_mask)['logits']
      masked_idx=(input_ids == tokenizer.mask_token_id)[0]
      scores=output_logits[:,masked_idx,:].squeeze() # scores (for whole vocabulary) at masked position in P(x) (pattern)

      score_pos,score_neg=scores[:,self.id_pos],scores[:,self.id_neg]
      restr_scores=torch.cat((score_pos,score_neg),dim=1) # scores of v(1) and v(0) at masked pos
      prob=F.softmax(restr_scores, dim=1)
      return prob[:,0] # as binary classif, return proba of each ex to be pos (other one is just 1 - ...)
    



if __name__ == "__main__":

    n_samples=1000
    raw_data = load_dataset("scikit-learn/imdb", split="train")
    raw_data=raw_data.shuffle()
    # model_name = "distilbert-base-uncased" 
    model_name = "albert-base-v2" 
    # tokenizer=DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    tokenizer=AlbertTokenizer.from_pretrained(model_name, do_lower_case=True)
    pvp=PVP(1,1)

    # prepare specific pvp data
    dataset_11=CustomDataset(tokenizer,raw_data,pvp,n_samples=n_samples)
    # train/test split
    examples_11, test_11 = Subset(dataset_11, range(32)),Subset(dataset_11, range(32, len(dataset_11)))
    # set up loaders
    data_collator = DataCollator(dataset_11.tokenizer)
    bsize=8
    examples_11_loader = DataLoader(examples_11, batch_size=bsize, collate_fn=data_collator) # train for 1 batch
    test_11_loader = DataLoader(test_11, batch_size=16, collate_fn=data_collator)

    # prepare model 
    # mlm = DistilBertForMaskedLM.from_pretrained(model_name)
    mlm = AlbertForMaskedLM.from_pretrained(model_name)
    model=Classifier(mlm,tokenizer,pvp)

    # PET
    train_loss, train_acc, test_loss, test_acc=PET(model, examples_11_loader, test_11_loader,bsize=bsize, lr=1e-5)

        

    
