import numpy as np
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

def get_save_path(args,num_model):
    save_path='models/'+args.model+'/'
    if (args.id_p is not None) and (args.id_v is not None):
        save_path=save_path+'pet_pvp='+str(args.id_p)+str(args.id_v)+'/'
    else:
        save_path=save_path+'classic/'
    if num_model is None:
        s=''
    else:
        s=str(num_model)
    save_path=save_path+'model'+s+'.pth'
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
       model_name="albert-base-v2"
       tokenizer=AlbertTokenizer.from_pretrained(model_name, do_lower_case=True) 
    return model_name, tokenizer

def create_backbone(args,model_name,tokenizer,pvp):
    model_prefix=args.model
    if (args.id_p is not None) and (args.id_v is not None) : 
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

def preprocessing_fn(x, tokenizer,pvp):
    transformed_data={}
    x["label"] = 0 if x["sentiment"] == "negative" else 1
    transformed_data["labels"]=x["label"]
    if pvp is not None:
        pattern,verbalized_label=pvp.transform_data(x["review"],x["label"])
        transformed_data["input_ids"] = tokenizer(
            pattern,
            add_special_tokens=False,
            truncation=True,
            max_length=511,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        # !!! NO CLS AND SEP (?): we use the score yielded by the repr of the <MASK> token (cf paper), that is we take
        # the embedding of the elements at the i* th position (masked idx) and we use it to further calculate score.
        transformed_data["input_ids"]=transformed_data["input_ids"]+[tokenizer.mask_token_id]
        # transformed_data["input_ids"]=[tokenizer.cls_token_id]+transformed_data["input_ids"]+[tokenizer.mask_token_id,tokenizer.sep_token_id]
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
    return transformed_data
       

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # `batch` is a list of dictionary with keys "review_ids" and "label".
        features = self.tokenizer.pad(
            batch, padding="longest", max_length=512, return_tensors="pt"
        )
        return features
    



# MODEL RELATED METHODS
    
def train(model, examples_loader, test_loader, bsize=32,lr=1e-4,eval=False,save_path=None):
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
    n_iter=32//bsize # to do 1 epoch and exactly 32 examples
    for k in range(n_iter):
        # ========== Training (32 examples) ==========
        # Set model to training mode
        model.train()
        model.to(DEVICE)
        train_loss = 0
        epoch_train_acc = 0
        for batch in tqdm(examples_loader): # a single 32 batch in examples_loader
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
        
        if k==n_iter-1: # at the end
            if eval:
                # ========== test ==========
                l, a = test(model, test_loader)
                print("Final :",
                    "\ntrain loss: {:.4f}".format(list_train_loss[-1]),
                    "train acc: {:.4f}".format(list_train_acc[-1]),
                    "test loss: {:.4f}".format(l),
                    "test acc:{:.4f}".format(a * 100),
                )
        else:
            
            print(k,
            "\ntrain loss: {:.4f}".format(list_train_loss[-1]),
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
      acc=(preds.squeeze()>0.5)==labels
      total_size+=acc.shape[0]
      acc_total+=acc.sum().item()
      loss_total+=loss.item()
  return loss_total/len(test_dataloader),acc_total/total_size


def test_ensemble(list_models,test_dataloader):
  total_size=0
  acc_total=0
  loss_total=0
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
      list_preds=[]
      for model in list_models:
          preds=model(input_ids=input_ids,attention_mask=attention_mask)
          list_preds.append(preds)
      global_pred = torch.stack(list_preds)
      global_pred= torch.mean(global_pred, dim=0)
      loss=criterion(global_pred.squeeze(),labels)
      acc=(global_pred.squeeze()>0.5)==labels
      total_size+=acc.shape[0]
      acc_total+=acc.sum().item()
      loss_total+=loss.item()
  return loss_total/len(test_dataloader),acc_total/total_size

