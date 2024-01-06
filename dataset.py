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
    def __init__(self,tokenizer,raw_data,pvp,n_samples):
        self.tokenizer= tokenizer
        self.raw_data=raw_data
        self.pvp=pvp
        self.dataset=self.create_new_data(n_samples)
        pass



    def create_new_data(self,n_samples):
        samples=self.raw_data.shuffle().select(range(n_samples))
        tokenized=samples.map(preprocessing_fn , fn_kwargs={"tokenizer":self.tokenizer, "pvp":self.pvp})
        to_keep=["input_ids","target","labels"]
        tokenized = tokenized.select_columns(to_keep)
        return tokenized

    def observe(self,idx_test):
        # idx_test: a single index
        dic=self.__getitem__(idx_test)
        pattern_ids,answer_id=dic['input_ids'],dic['target']
        pattern_decoded = self.tokenizer.decode(pattern_ids, skip_special_tokens=False)
        answer_decoded = self.tokenizer.decode(answer_id, skip_special_tokens=False)
        return pattern_decoded, answer_decoded
    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        sample = {'input_ids': self.dataset[idx]['input_ids'], 'target': self.dataset[idx]['target']\
                  , 'labels':self.dataset[idx]['labels']}
        return sample
    

if __name__ == "__main__":

    n_samples=1000
    raw_data = load_dataset("scikit-learn/imdb", split="train")
    raw_data=raw_data.shuffle()
    tokenizer=DistilBertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    pvp=PVP(1,1)
    dataset_11=CustomDataset(tokenizer,raw_data,pvp,n_samples=n_samples)
    
    # test dataset is correct
    pattern_decoded, answer_decoded=dataset_11.observe(0)
    print(pattern_decoded)
    print(answer_decoded)

    # train/test split
    examples_11, test_11 = Subset(dataset_11, range(32)),Subset(dataset_11, range(32, len(dataset_11)))

    # eventually set up loaders ?
    data_collator = DataCollator(dataset_11.tokenizer)
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
    

