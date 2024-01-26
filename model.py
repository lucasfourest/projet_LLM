import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, Subset
from datasets import load_dataset
from tqdm.autonotebook import *
from transformers import AlbertForMaskedLM, AlbertTokenizer, BertForMaskedLM, BertTokenizer, \
  DistilBertForMaskedLM, DistilBertTokenizer, AlbertModel, BertModel, DistilBertModel

from pvp import *
from dataset import *
from utils import *


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self, bert):
      super(Classifier, self).__init__()
      self.bert=bert
      self.emb_dim=self.bert.config.hidden_size
      self.classifier=nn.Sequential(
          nn.Dropout(0.3),
          nn.Linear(self.emb_dim,1)
      )

    def forward(self, input_ids, attention_mask, **kwargs):
      bert_output=self.bert(input_ids=input_ids,attention_mask=attention_mask)
      cls=bert_output["last_hidden_state"][:,0,:]
      prob=F.sigmoid(self.classifier(cls))
      return prob
    


class PETClassifier(nn.Module):
    def __init__(self, mlm,tokenizer,pvp):
      super(PETClassifier, self).__init__()
      self.mlm = mlm # "masked language model"
      self.tokenizer = tokenizer
      self.id_pos, self.id_neg=self.tokenizer.encode(pvp.verbalizer.answer_pos,add_special_tokens=False,),\
                                self.tokenizer.encode(pvp.verbalizer.answer_neg,add_special_tokens=False,)

    def get_scores(self, input_ids, attention_mask):
       output_logits=self.mlm(input_ids=input_ids,attention_mask=attention_mask)['logits']
       masked_idx=(input_ids == self.tokenizer.mask_token_id)[0]
       scores=output_logits[:,masked_idx,:].squeeze() # scores (for whole vocabulary) at masked position in P(x) (pattern)
       score_pos,score_neg=scores[:,self.id_pos],scores[:,self.id_neg]
       restr_scores=torch.cat((score_pos,score_neg),dim=1)
       return restr_scores

    def forward(self, input_ids, attention_mask):
      restr_scores=self.get_scores(input_ids, attention_mask) # scores of v(1) and v(0) at masked pos
      prob=F.softmax(restr_scores, dim=1)
      to_return=prob[:,0]
      return to_return # as binary classif, return proba of each ex to be pos (other one is just 1 - ...)
    




    
