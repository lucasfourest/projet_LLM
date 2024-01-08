import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, Subset
from datasets import load_dataset
from tqdm.autonotebook import *
from transformers import AlbertModel, AlbertTokenizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    
def train(model, examples_loader, test_loader, bsize=32,lr=1e-4,validation=False,save=False):
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
            # ========== test ==========
            l, a = test(model, test_loader)
            print("Final :",
                "\ntrain loss: {:.4f}".format(list_train_loss[-1]),
                "train acc: {:.4f}".format(list_train_acc[-1]),
                "test loss: {:.4f}".format(l),
                "test acc:{:.4f}".format(a * 100),
            )
        else:
            if validation:
                # ========== valid ==========
                l, a = test(model, test_loader)
                print(k,
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