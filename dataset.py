import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader, Subset
from datasets import load_dataset
from tqdm.autonotebook import *
from transformers import AlbertForMaskedLM, AlbertTokenizer, BertForMaskedLM, BertTokenizer, \
  DistilBertForMaskedLM, DistilBertTokenizer

from pvp import *
from utils import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class CustomDataset(Dataset):
    def __init__(self,tokenizer,raw_data,pvp=None,n_samples=5000):
        self.tokenizer= tokenizer
        self.raw_data=raw_data
        self.pvp=pvp
        self.dataset=self.create_new_data(n_samples)
        pass



    def create_new_data(self,n_samples):
        samples=self.raw_data.shuffle().select(range(n_samples))
        tokenized=samples.map(preprocessing_fn , fn_kwargs={"tokenizer":self.tokenizer, "pvp":self.pvp})
        if self.pvp is not None : to_keep=["input_ids","target","labels"]
        else : to_keep=["input_ids","labels"]
        tokenized = tokenized.select_columns(to_keep)
        return tokenized

    def observe(self,idx_test):
        # idx_test: a single index
        dic=self.__getitem__(idx_test)
        if self.pvp is not None:
            pattern_ids,answer_id=dic['input_ids'],dic['target']
            pattern_decoded = self.tokenizer.decode(pattern_ids, skip_special_tokens=False)
            answer_decoded = self.tokenizer.decode(answer_id, skip_special_tokens=False)
            return pattern_decoded, answer_decoded
        else:
            input_decoded = self.tokenizer.decode(dic['input_ids'], skip_special_tokens=False)
            label = dic['labels']
            return input_decoded, label

    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        if self.pvp is not None:
            sample = {'input_ids': self.dataset[idx]['input_ids'], 'target': self.dataset[idx]['target']\
                  , 'labels':self.dataset[idx]['labels']}
        else:
            sample={'input_ids': self.dataset[idx]['input_ids'], 'labels':self.dataset[idx]['labels']}
        return sample
    

if __name__ == "__main__":

    n_samples=1000
    raw_data = load_dataset("scikit-learn/imdb", split="train")
    raw_data=raw_data.shuffle()
    tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
    dataset=CustomDataset(tokenizer,raw_data,pvp=None,n_samples=n_samples) # normal dataset
    pvp=PVP(1,1)
    dataset_11=CustomDataset(tokenizer,raw_data,pvp=pvp,n_samples=n_samples)
    
    # test if both datasets are correct
    print('-'*25 + 'Normal Dataset'+'-'*25 )
    input_decoded, label=dataset.observe(0)
    print(input_decoded)
    print(label)
    print('-'*25 + 'Custom Dataset'+'-'*25 )
    pattern_decoded, answer_decoded=dataset_11.observe(0)
    print(pattern_decoded)
    print(answer_decoded)

    # train/test split
    examples, test = Subset(dataset, range(32)),Subset(dataset, range(32, len(dataset)))
    examples_11, test_11 = Subset(dataset_11, range(32)),Subset(dataset_11, range(32, len(dataset_11)))

    # Set up loaders 
    # Normal one
    print('-'*25 + 'Normal Dataloader'+'-'*25 )
    data_collator = DataCollator(dataset.tokenizer)
    examples_loader = DataLoader(examples, batch_size=32, collate_fn=data_collator) # train for 1 batch
    test_loader = DataLoader(test, batch_size=256, collate_fn=data_collator)
    batch = next(iter(examples_loader))
    print(f'input ids: {batch["input_ids"][0]}')
    print(f"attention mask: {batch['attention_mask'][0]}")
    print(f"label: {batch['labels'][0]}")
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    # Custom one
    print('-'*25 + 'Custom Dataloader'+'-'*25 )
    examples_11_loader = DataLoader(examples_11, batch_size=32, collate_fn=data_collator) # train for 1 batch
    test_11_loader = DataLoader(test_11, batch_size=256, collate_fn=data_collator)
    batch = next(iter(examples_11_loader))
    print(f'input ids: {batch["input_ids"][0]}')
    print(f"attention mask: {batch['attention_mask'][0]}")
    print(f"target id: {batch['target'][0]}")
    print(f"label: {batch['labels'][0]}")
    print(batch['input_ids'].shape)
    print(batch['labels'].shape)
    print(batch['target'].shape)
    

