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
    



if __name__ == "__main__":

    n_samples=1000
    raw_data = load_dataset("scikit-learn/imdb", split="train")
    raw_data=raw_data.shuffle()
    model_name = "distilbert-base-uncased" 
    # model_name = "bert-base-uncased" 
    # model_name = "albert-base-v1" 
    tokenizer=DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    # tokenizer=BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    # tokenizer=AlbertTokenizer.from_pretrained(model_name, do_lower_case=True)
    pvp=PVP(1,1)


    # prepare data
    dataset=CustomDataset(tokenizer,raw_data,pvp=None,n_samples=n_samples) # normal dataset
    dataset_11=CustomDataset(tokenizer,raw_data,pvp,n_samples=n_samples) # specific pvp dataset
    # train/test split
    examples, test = Subset(dataset, range(32)),Subset(dataset, range(32, len(dataset)))
    examples_11, test_11 = Subset(dataset_11, range(32)),Subset(dataset_11, range(32, len(dataset_11)))
    # set up loaders
    data_collator = DataCollator(dataset_11.tokenizer)
    bsize=8
    examples_loader = DataLoader(examples, batch_size=bsize, collate_fn=data_collator) # train for 1 batch
    test_loader = DataLoader(test, batch_size=32, collate_fn=data_collator)
    examples_11_loader = DataLoader(examples_11, batch_size=bsize, collate_fn=data_collator) # train for 1 batch
    test_11_loader = DataLoader(test_11, batch_size=32, collate_fn=data_collator)


    # prepare normal classifier model 
    bert = DistilBertModel.from_pretrained(model_name)
    # bert = BertModel.from_pretrained(model_name)
    # bert = AlbertModel.from_pretrained(model_name)
    model=Classifier(bert)
    print('\n'+'-'*25 + 'Normal Training'+'-'*25 )
    train(model, examples_loader, test_loader,bsize=bsize, lr=1e-5) # normal training with 32 examples
    

    # prepare PET model 
    mlm = DistilBertForMaskedLM.from_pretrained(model_name)
    # mlm = BertForMaskedLM.from_pretrained(model_name)
    # mlm = AlbertForMaskedLM.from_pretrained(model_name)
    model=PETClassifier(mlm,tokenizer,pvp)
    print('\n'+'-'*25 + 'PE Training'+'-'*25 )
    train(model, examples_11_loader, test_11_loader,bsize=bsize, lr=1e-6) # PET with 32 examples     

    
