'''
Fine-tuning CodeBERT based on bug reports and projects.
'''

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load Dataset
df = pd.read_csv('../data/dataset_codebert.csv')

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Chunk size for source code chunks
CHUNK_SIZE = 512

def chunk_source_code(source_code, max_length=CHUNK_SIZE, overlap=50):
    '''
    Split source code into overlapping chunks.
    '''
    tokens = tokenizer.tokenize(source_code)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i : 1 + max_length]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

class BugLocalizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        bug_report = str(row['bug_report'])
        source_code = str(row['source_code'])
        label = int(row['label'])

        # Create source chunks and choose the first chunk
        src_chunks = chunk_source_code(source_code, max_length=self.max_length)
        if not src_chunks:
            return None
        src_chunk = src_chunks[0] # Using the first chunk here

        inputs = self.tokenizer(
            bug_report, 
            src_chunk, 
            truncation=True,
            padding="max_length",
            max_length = self.max_length,
            return_tensors = "pt"
            )
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        return inputs
    
# Create dataset and dataloader
dataset = BugLocalizationDataset(df, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
