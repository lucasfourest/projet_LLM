import numpy as np
import regex as re
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, Subset
from datasets import load_dataset
from tqdm.autonotebook import *
from transformers import AlbertForMaskedLM, AlbertTokenizer, BertForMaskedLM, BertTokenizer, \
  DistilBertForMaskedLM, DistilBertTokenizer, AlbertModel, BertModel, DistilBertModel



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PIPELINE

def load_raw_data(ds_name):
    if ds_name=='imdb':
        raw_data=load_dataset("scikit-learn/imdb", split="train")
    if ds_name=='boolq':
        raw_data=load_dataset("super_glue", "boolq", split='train')
    return raw_data

def get_save_path(args):
    save_path='models/'+args.model+'/'+args.dataset+'/'
    if (args.id_p is not None) and (args.id_v is not None):
        save_path=save_path+'pet_pvp='+str(args.id_p)+str(args.id_v)+'/'
    else:
        save_path=save_path+'classic/'
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    save_path=save_path+'model.pth'
    return save_path

    

def create_name_tokenizer(args):
    model_prefix=args.model
    if model_prefix =='distilbert': 
       model_name="distilbert-base-uncased"
       tokenizer=DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_prefix =='bert': 
       model_name="bert-base-uncased"
       tokenizer=BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    if model_prefix=='albert':
       model_name="albert-large-v2"
       tokenizer=AlbertTokenizer.from_pretrained(model_name, do_lower_case=True) 
    return model_name, tokenizer

def create_backbone(args,model_name,id_p=None, id_v=None):
    model_prefix=args.model
    if (id_p is not None) and (id_v is not None) : 
        if model_prefix=='distilbert':mlm = DistilBertForMaskedLM.from_pretrained(model_name)
        if model_prefix=='bert': mlm= BertForMaskedLM.from_pretrained(model_name)
        if model_prefix=='albert':mlm = AlbertForMaskedLM.from_pretrained(model_name)
        backbone=mlm
        
    else :     
        if model_prefix=='distilbert':bert = DistilBertModel.from_pretrained(model_name)
        if model_prefix=='bert':bert = BertModel.from_pretrained(model_name)
        if model_prefix=='albert':bert = AlbertModel.from_pretrained(model_name)
        backbone=bert

    return backbone


# DATA PROCESSING

def process_text(txt):
    x = re.sub('[,\.!?:()"]', '', txt)
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http\S+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = re.sub('\s+', ' ', x)
    x=x.lower().strip()
    return x

def preprocessing_func(x, tokenizer,pvp, name ='imdb'):
    transformed_data={}
    if name=='imdb':
        x["label"] = 0 if x["sentiment"] == "negative" else 1
        transformed_data["labels"]=x["label"]
        inputs=[x['review']]
        if pvp is not None:
            [before_mask,after_mask],verbalized_label=pvp.transform_data(inputs,x["label"])
            before_tok = tokenizer(
                before_mask,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            after_tok = tokenizer(
                before_mask,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            # !!! NO CLS AND SEP (?): we use the score yielded by the repr of the <MASK> token (cf paper), that is we take
            # the embedding of the elements at the i* th position (masked idx) and we use it to further calculate score.
            if len(before_tok + after_tok) >511:
                if len(before_tok)>511:
                    transformed_data["input_ids"]=before_tok[:511]+[tokenizer.mask_token_id]
                else:
                    transformed_data["input_ids"]=before_tok+[tokenizer.mask_token_id]
            else:
                transformed_data["input_ids"]=before_tok+[tokenizer.mask_token_id] +after_tok
            
            transformed_data["target"]=tokenizer(
                verbalized_label,
                add_special_tokens=False,
                truncation=True,
                max_length=512,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
        else:
            transformed_data["input_ids"] = tokenizer(
            x["review"],
            add_special_tokens=False,
            truncation=True,
            max_length=511,
            padding=False,
            return_attention_mask=False,
            )["input_ids"]
            # In normal classification mode, the cls token at beginning is needed!
            transformed_data["input_ids"]=[tokenizer.cls_token_id]+transformed_data["input_ids"]
    if name=='boolq':
        transformed_data["labels"]=x["label"]
        inputs=[x['passage'], x['question']]
        if pvp is not None:
            [before_mask,after_mask],verbalized_label=pvp.transform_data(inputs,x["label"])
            before_tok = tokenizer(
                before_mask,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            after_tok = tokenizer(
                after_mask,
                add_special_tokens=False,
                truncation=True,
                max_length=511,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            # !!! NO CLS AND SEP (?): we use the score yielded by the repr of the <MASK> token (cf paper), that is we take
            # the embedding of the elements at the i* th position (masked idx) and we use it to further calculate score.
            if len(before_tok + after_tok) >511:
                if len(before_tok)>511:
                    transformed_data["input_ids"]=before_tok[:511]+[tokenizer.mask_token_id]
                else:
                    transformed_data["input_ids"]=before_tok+[tokenizer.mask_token_id]
            else:
                transformed_data["input_ids"]=before_tok+[tokenizer.mask_token_id] +after_tok
            
            transformed_data["target"]=tokenizer(
                verbalized_label,
                add_special_tokens=False,
                truncation=True,
                max_length=512,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
        else:
            transformed_data["input_ids"] = tokenizer(
            x['passage']+ ' ' +x['question'],
            add_special_tokens=False,
            truncation=True,
            max_length=511,
            padding=False,
            return_attention_mask=False,
            )["input_ids"]
            # In normal classification mode, the cls token at beginning is needed!
            transformed_data["input_ids"]=[tokenizer.cls_token_id]+transformed_data["input_ids"]
    return transformed_data
       

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        features = self.tokenizer.pad(
            batch, padding="longest", max_length=512, return_tensors="pt"
        )
        return features
    



# MODEL RELATED METHODS
    
def train(model, examples_loader, test_loader,n_epochs=1,lr=1e-4,eval_every=None,save_path=None):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-08,
    )
    list_test_acc = []
    list_train_acc = []
    list_train_loss = []
    list_test_loss = []
    criterion = nn.BCELoss()
    for epoch in range(n_epochs):
        # ========== Training ==========
        # Set model to training mode
        model.train()
        model.to(DEVICE)
        train_loss = 0
        epoch_train_acc = 0
        for batch in tqdm(examples_loader): 
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )

            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Backward pass
            loss = criterion(outputs.squeeze(), labels.squeeze().float())
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()
            acc = (outputs.squeeze() > 0.5) == labels.squeeze()
            epoch_train_acc += acc.float().mean().item()
        list_train_acc.append(100 * epoch_train_acc / len(examples_loader))
        list_train_loss.append(train_loss / len(examples_loader))
    
        if eval_every is not None:
            if (epoch+1)%eval_every==0:
                # ========== test ==========
                l, a = test(model, test_loader)
                list_test_acc.append(a)
                list_test_loss.append(l)
                print(epoch,
                    "train loss: {:.4f}".format(list_train_loss[-1]),
                    "train acc: {:.4f}".format(list_train_acc[-1]),
                    "test loss: {:.4f}".format(l),
                    "test acc:{:.4f}".format(a * 100),
                )
            else:
                
                print(epoch,
                "train loss: {:.4f}".format(list_train_loss[-1]),
                "train acc: {:.4f}".format(list_train_acc[-1]),
                )
        else:
                
                print(epoch,
                "train loss: {:.4f}".format(list_train_loss[-1]),
                "train acc: {:.4f}".format(list_train_acc[-1]),
                )

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc


def test(model,test_dataloader):
  total_size=0
  acc_total=0
  loss_total=0
  criterion=nn.BCELoss()
  model.eval()
  with torch.no_grad():
    for batch in tqdm(test_dataloader):
      batch={k:v.to(DEVICE) for k,v in batch.items()}
      input_ids=batch["input_ids"]
      labels=batch["labels"]
      attention_mask=batch["attention_mask"]
      labels=labels.float()
      preds=model(input_ids=input_ids,attention_mask=attention_mask)
      loss=criterion(preds.squeeze(),labels)
      acc=(preds.squeeze()>0.5)==labels.squeeze()
      total_size+=acc.shape[0]
      acc_total+=acc.sum().item()
      loss_total+=loss.item()
  return loss_total/len(test_dataloader),acc_total/total_size


def test_ensemble(list_models,list_weigths,test_dataloader):
  total_size=0
  acc_total=0
  loss_total=0
  weights_tensor=torch.tensor(list_weigths).to(DEVICE)
  criterion=nn.BCELoss()
  for model in list_models:
      model.eval()
  with torch.no_grad():
    for batch in tqdm(test_dataloader):
      batch={k:v.to(DEVICE) for k,v in batch.items()}
      input_ids=batch["input_ids"]
      labels=batch["labels"]
      attention_mask=batch["attention_mask"]
      labels=labels.float()
      
      global_pred=make_ensemble_pred(list_models,weights_tensor, input_ids, attention_mask, labels)
      loss=criterion(global_pred.squeeze(),labels)
      acc=(global_pred.squeeze()>0.5)==labels.squeeze()
      total_size+=acc.shape[0]
      acc_total+=acc.sum().item()
      loss_total+=loss.item()
  return loss_total/len(test_dataloader),acc_total/total_size


def make_ensemble_pred(list_models,weights_tensor, input_ids, attention_mask, labels):

    
    list_scores_pos=[]
    list_scores_neg=[]
    for model in list_models:
        scores_pos_neg=model.get_scores(input_ids=input_ids,attention_mask=attention_mask)
        scores_pos, scores_neg=scores_pos_neg[:,0],scores_pos_neg[:,1]
        list_scores_pos.append(scores_pos),list_scores_neg.append(scores_neg)
    global_scores_pos,global_scores_neg = torch.stack(list_scores_pos),torch.stack(list_scores_neg)
    weighted_pos=weights_tensor[:,None]*global_scores_pos
    weighted_neg=weights_tensor[:,None]*global_scores_neg
    w_sum_pos, w_sum_neg=torch.sum(weighted_pos,dim=0),torch.sum(weighted_neg,dim=0)
    w_sum=torch.cat((w_sum_pos.unsqueeze(-1), w_sum_neg.unsqueeze(-1)), dim=1)
    probs=F.softmax(w_sum, dim=1)
    global_pred=probs[:,0]
    return global_pred

