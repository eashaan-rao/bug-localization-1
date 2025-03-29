'''
Fine-tuning CodeBERT based on bug reports and projects.
'''

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import transformers
import logging


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Chunk size for source code chunks
CHUNK_SIZE = 512
MAX_BUG_REPORT_TOKENS = 256
MAX_SRC_CHUNK_TOKENS = 256

def chunk_source_code(source_code, bug_report_tokens, max_length=CHUNK_SIZE, overlap=50):
    '''
    Split source code into overlapping chunks with space for bug report tokens.
    '''
    # Reserve space for bug report + special tokens
    chunk_size = max(max_length - len(bug_report_tokens) - 10, 1)

    # tokenize and split source code into chunks
    tokens = tokenizer.tokenize(source_code)

    # Ensure the step is valid to prevent range errors
    effective_step = max(chunk_size - overlap, 1)

    chunks = []
    for i in range(0, len(tokens), effective_step):
        chunk = tokens[i : i + chunk_size]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks

class BugLocalizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Precompute chunks and store them in self.samples
        for _, row in df.iterrows():
            bug_report = str(row['bug_report'])
            source_code = str(row['source_code'])
            label = int(row['label'])

            # Tokenize bug report to calculate available space
            bug_report_tokens = tokenizer.tokenize(bug_report)
            if len(bug_report_tokens) > MAX_BUG_REPORT_TOKENS:
                bug_report_tokens =  bug_report_tokens[:128] + bug_report_tokens[-128:]
                max_length = MAX_SRC_CHUNK_TOKENS
            else:
                # Allow dynamic length for source chunk if bug report is shorter
                max_length = self.max_length - len(bug_report_tokens) - 10

            # generate all chunks
            src_chunks = chunk_source_code(source_code, bug_report_tokens, max_length=max_length)

            # Store all (bug_report, src_chunk, label) pairs
            for src_chunk in src_chunks:
                self.samples.append((bug_report, src_chunk, label))


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        '''
            Returns (bug_report, src_chunk, label) as input for CodeBERT.
        '''
        bug_report, src_chunk, label = self.samples[idx]

        # Prepare inputs for the model
        inputs = self.tokenizer(
            bug_report, 
            src_chunk, 
            truncation=True,
            padding="max_length",
            max_length = self.max_length,
            return_tensors = "pt"
            )
        inputs["labels"] = torch.tensor(label, dtype=torch.long)

        return {key: val.squeeze(0) for key, val in inputs.items() if val is not None}

def codebert_finetune():

    #Load dataset
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    # Create the models/codebert_finetune_EPUI folder inside data
    models_path = os.path.join(data_folder_path, "models/codebert_finetune_EPUI")
    os.makedirs(models_path, exist_ok=True)
    file_path = os.path.join(data_folder_path, "dataset_codebert.csv")
    df = pd.read_csv(file_path)

    # Split dataset into train/val
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Create dataset and dataloaders
    train_dataset = BugLocalizationDataset(train_df, tokenizer)
    val_dataset = BugLocalizationDataset(val_df, tokenizer)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=models_path,
        num_train_epochs=3,
        per_device_eval_batch_size=8,
        per_device_train_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        disable_tqdm=False,  # Enable progress bar
        fp16=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Enable progress bar explicitly
    transformers.utils.logging.enable_progress_bar()
    transformers.utils.logging.set_verbosity_info()
    logging.getLogger("transformers").setLevel(logging.INFO)
    
    #Define Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        data_collator = data_collator
    )

    # Start Fine-tuning
    trainer.train()

    # Save fine-tune model
    model.save_pretrained(models_path)
    tokenizer.save_pretrained(models_path)
    print("Fine-tuning complete! Model saved.")


