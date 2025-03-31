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
import time
from datetime import datetime  # Import the datetime module
import warnings
from transformers import logging

# suppress transfromers warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned for the setting you have chosen, i.e. sequence pairs with the 'longest_first' truncation strategy. So the returned list will always be empty even if some tokens have been removed.")
logging.get_logger("transformers").setLevel(logging.ERROR) # also set the logging level for the transformers module itself.

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

CHUNK_SIZE = 512
MAX_BUG_REPORT_TOKENS = 256
MAX_SRC_CHUNK_TOKENS = 256

def chunk_source_code(source_code, max_length, overlap=50):
    '''
    Split source code into overlapping chunks with specified max_length.
    '''
    # tokenize and split source code into chunks
    tokens = tokenizer.tokenize(source_code)
    # Ensure the step is valid to prevent range errors
    effective_step = max(max_length - overlap, 1)

    chunks = []
    for i in range(0, len(tokens), effective_step):
        chunk = tokens[i : i + max_length]
        chunks.append(tokenizer.convert_tokens_to_ids(chunk))  #convert tokens to ids.
    return chunks

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=8,
            num_workers=0,  # Utilize CPU Cores
            pin_memory=True   # Faster GPU Transfer
        )

    def get_eval_dataloader(self, eval_dataset = None):
        return DataLoader(
            self.eval_dataset if eval_dataset is None else eval_dataset,
            batch_size=8,
            num_workers=0,  # Utilize CPU Cores
            pin_memory=True   # Faster GPU Transfer
        )   

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

            # Tokenize bug report and truncate if necessary to calculate available space for src_code chunk
            bug_report_ids = tokenizer.encode(bug_report, add_special_tokens=False)
            if len(bug_report_ids) > MAX_BUG_REPORT_TOKENS:
                bug_report_ids = bug_report_ids[:128] + bug_report_ids[-128:]
                # bug_report = tokenizer.decode(bug_report_ids) # Re-decode truncated bug report

            # Calculate remaining space for source code chunks
            max_src_chunk_length = self.max_length - len(bug_report_ids) - 3  # Account for special tokens

            # Checking
            # print(f"Bug report token IDs Length: {len(bug_report_ids)}")
            # print(f"final bug report length: {len(bug_report)}")
            # print(f"max_src_chunk_length: {max_src_chunk_length}")

            # generate source code chunks
            src_chunks = chunk_source_code(source_code, max_src_chunk_length)

            # Store all (bug_report, src_chunk, label) pairs
            for src_chunk in src_chunks:
                self.samples.append((bug_report_ids, src_chunk, label))


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        '''
            Returns (bug_report, src_chunk, label) as input for CodeBERT.
        '''
        bug_report_ids, src_chunk, label = self.samples[idx]

        # Prepare inputs for the model
        inputs = self.tokenizer.prepare_for_model(
            bug_report_ids, 
            src_chunk, 
            truncation="longest_first",     # Prioritize longest sequence
            padding= 'max_length',
            max_length = self.max_length,
            return_tensors = "pt"
            )
        inputs["labels"] = torch.tensor(label, dtype=torch.long)

        # print(f"Sample {idx}:")
        # print(f"  input_ids shape: {inputs['input_ids'].shape}")
        # print(f"  attention_mask shape: {inputs['attention_mask'].shape}")
        # print(f"  labels shape: {inputs['labels'].shape}")
        # print(f"  input_ids length: {len(inputs['input_ids'][0])}") #print the length of the input_ids.

        return {key: val.squeeze(0) for key, val in inputs.items() if val is not None}

def codebert_finetune():
    start_time = time.time() 
    start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') #convert to string.
    print(f"Start time: {start_time_formatted}") #print the start time.

    #Load dataset
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    # Create the models/codebert_finetune_EPUI folder inside data
    models_path = os.path.join(data_folder_path, "models/codebert_finetune_EPUI")
    os.makedirs(models_path, exist_ok=True)
    file_path = os.path.join(data_folder_path, "dataset_codebert.csv")
    df = pd.read_csv(file_path)

    # num_rows = len(df)
    # print(f"Number of rows in dataset_codebert.csv: {num_rows}")

    # Limit to 100 rows
    df = df.head(100000)

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
        fp16=True,
        dataloader_num_workers=0, # Utilize CPU cores during data loading
        gradient_accumulation_steps=2  # Larger effective batch size
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Enable progress bar explicitly
    transformers.utils.logging.enable_progress_bar()
    # transformers.utils.logging.set_verbosity_info()
    # logging.get_logger("transformers").setLevel(logging.INFO)
    
    #Define Trainer
    trainer = CustomTrainer(
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

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Fine-tuning took {elapsed_time:.2f} seconds.") #print the elapsed time


