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
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm
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
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,  # Utilize CPU Cores
            pin_memory=True   # Faster GPU Transfer
        )

    def get_eval_dataloader(self, eval_dataset = None):
        return DataLoader(
            self.eval_dataset if eval_dataset is None else eval_dataset,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,  # Utilize CPU Cores
            pin_memory=True   # Faster GPU Transfer
        )   

class BugLocalizationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        #  A list to map each created sample back to its metadata
        self.sample_to_metadata = []
        self.df = df # Store df to access metadata later
        
        # Precompute chunks and store them in self.samples
        for _, row in df.iterrows():
            bug_report = str(row['bug_report'])
            source_code = str(row['source_code'])
            label = int(row['label'])
            # We need the filename to map back to later
            file_name = str(row['file_name'])

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
                # For every sample, store its original filename and label
                self.sample_to_metadata.append({'file_name': file_name, 'label': label})


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

# ADDED New: Function to predict scores and rank files for each bug report
def predict_and_rank_for_codebert(trainer, test_df):
    model.eval()
    bug_report_groups = test_df.groupby('bug_report')
    bug_report_to_ranked_results = defaultdict(list)

    for bug_report, group  in tqdm(bug_report_groups, desc="Predicting and Ranking Test set"):
        # Create a temporary dataset for the current bug report files
        bug_report_dataset = BugLocalizationDataset(group, tokenizer)
        if not bug_report_dataset:
            continue

        # Get predictions (logits) from the trainer
        predictions = trainer.predict(bug_report_dataset)
        logits = predictions.predictions
        # Use the logit for the "positive" class (label 1) as the score
        scores = logits[:, 1]

        # Since one file can have multiple chunks, we average the scores
        file_scores = defaultdict(list)
        for i, score in enumerate(scores):
            # We need to map the sample index back to the original file name
            # original_index = bug_report_dataset.df.index[i]
            # FIX: Use the new metadata map to get the correct filename. This resolves the IndexError.
            metadata = bug_report_dataset.sample_to_metadata[i]
            filename = metadata['file_name']
            file_scores[filename].append(score)

        # Average scores for each file and get the label
        results = []
        for filename, score_list in file_scores.items():
            avg_score = np.mean(score_list)
            # Get the lavel from the first occurrence of the file
            label = group[group['file_name'] == filename]['label'].iloc[0]
            results.append({'filename': filename, 'label': label, 'score': avg_score})

        # Sort results by score in descending order
        bug_report_to_ranked_results[bug_report] = sorted(results, key=lambda x: x['score'], reverse=True)

    return bug_report_to_ranked_results

# ADDED new: function to compute final ranking metrics
def compute_ranking_metrics(bug_report_to_ranked_results, k_values=[1,5,10]):
    topk_accuracies = {k: [] for k in k_values}
    ap_scores = []
    rr_scores = []

    for bug_report, results in bug_report_to_ranked_results.items():
        ranked_labels = [item['label'] for item in results]
        scores = [item['score'] for item in results]

        # Top-K Accuracy
        for k in k_values:
            topk_accuracies[k].append(1 if 1 in ranked_labels[:k] else 0)

        # Mean Average Precision (MAP)
        if sum(ranked_labels) > 0:
            ap_scores.append(average_precision_score(ranked_labels, scores))
        
        # Mean Reciprocal Rank (MRR)
        try:
            first_hit_idx = ranked_labels.index(1)
            rr_scores.append(1.0/ (first_hit_idx + 1))
        except ValueError:
            rr_scores.append(0.0)

    topk = {k: np.mean(topk_accuracies[k]) if topk_accuracies[k] else 0 for k in k_values}
    map_score = np.mean(ap_scores) if ap_scores else 0
    mrr_score = np.mean(rr_scores) if rr_scores else 0

    return topk, map_score, mrr_score


def codebert_finetune(project_name):
    start_time = time.time() 
    start_time_formatted = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') #convert to string.
    print(f"Fine-tuning process started for project: '{project_name}' at {start_time_formatted}") #print the start time.

    #Load dataset
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')

    # Create project_specific paths for mode output and dataset input
    models_path = os.path.join(data_folder_path, "models", f"codebert_finetune_{project_name}")
    os.makedirs(models_path, exist_ok=True)

    dataset_filename = f"{project_name}_codebert_dataset.csv"
    file_path = os.path.join(data_folder_path, dataset_filename)

    print(f"Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at '{file_path}")

    df = pd.read_csv(file_path)
    print("Dataset reading complete.")

    # num_rows = len(df)
    # print(f"Number of rows in dataset_codebert.csv: {num_rows}")

    # Limit to 100 rows
    # df = df.head(100000)

    # Changed: Split dataset into train, validation and test sets (80/10/10 split) (Dataset Preparation)
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    # 0.11 of 0.9 is approx 0.1
    train_df, val_df = train_test_split(train_val_df, test_size=0.11, random_state=42, stratify=train_val_df['label'])

    # print("Available columns in test_df: ", test_df.columns)
    # exit(0)
    # Create dataset and dataloaders
    train_dataset = BugLocalizationDataset(train_df, tokenizer)
    val_dataset = BugLocalizationDataset(val_df, tokenizer)
    print(f"Dataset sizes: Train={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}")
    print("training dataset and validation dataset is created...")
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

    print("Trainer configured. Starting fine tuning...")
    print("Fine tuning starts now")

    # Start Fine-tuning
    trainer.train()
    print("Fine Tuning Complete...")

    # ADDED new: Final testing and evaluation phase
    print("\n Starting Final Evaluation on Test Set")
    # The trainer automatically loads the best model state when load_best_model_at_end=True
    ranked_results = predict_and_rank_for_codebert(trainer, test_df)
    topk_test, map_test, mrr_test = compute_ranking_metrics(ranked_results)

    print(f"\n CodeBERT Fine-tuned model performance for project: {project_name}... ")
    print(f"Mean Reciprocal Rank (MRR): {mrr_test:.4f}")
    print(f"Mean Average Precision (MAP): {map_test: .4f}")
    for k, acc in topk_test.items():
        print(f" Top-{k} Accuracy: {acc:.4f}")
    print('-'*40)

    # Save fine-tune model
    model.save_pretrained(models_path)
    tokenizer.save_pretrained(models_path)
    print(f"Fine-tuning complete for '{project_name}'!")
    print(f"Model saved to: {models_path}")

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Fine-tuning took {elapsed_time:.2f} seconds.") #print the elapsed time


