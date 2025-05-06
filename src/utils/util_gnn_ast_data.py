'''
Helper code to prepare dataset for GNN bug localization. 
'''

from torch_geometric.data import Data, InMemoryDataset
import torch
import logging
import os
import pickle
from tqdm import tqdm
import random
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%s(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Ast to PyG graph conversion
def ast_to_pyg_graph(ast_node):
    '''
       Convert AST (represented as nested dictionaries) to PyTorch Geometric graph

        Agrs:
            ast_node (dict): Root node of the AST
        Returns:
            Data: Pytorch geometric data object representing the graph
    '''
    node_features = []
    node_idx = 0
    edge_list = []

    def traverse(node, parent_idx=None):
        nonlocal node_idx
        current_idx = node_idx
        node_idx += 1

        # Create node feature
        label = node.get("type", "")
        text = node.get("text", "")
        feature_str = f"{label}_{text}" if text else label
        node_features.append(feature_str)

        # Add edge from parent (if not root)
        if parent_idx is not None:
            edge_list.append((parent_idx, current_idx))

        # Process children recursively
        for child in node.get("children", []):
            if child:  # Skip None children
                traverse(child, current_idx)
    
    # Build the graph structure
    traverse(ast_node)

    # Convert features to indices
    vocab = {feat: i for i, feat in enumerate(set(node_features))}
    x = torch.tensor([vocab[f] for f in node_features], dtype=torch.long).unsqueeze(1)

    # Create edge index tensor [2, num_edges]
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # Handle case with no edges (single node)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

# Dataset splitter (by bug report)
def split_bug_reports(all_bug_reports, ratios=(0.7, 0.15, 0.15), seed=42):
    '''
    Split bug reports into train, validation and test sets

    Args:
        all_bug_reports (list): List of bug report IDs
        ratios (tuple): Proportions for train, val, test splits
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_ids, val_ids, test_ids)
    '''
    random.seed(seed)
    random.shuffle(all_bug_reports)
    
    n = len(all_bug_reports)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    return (
        all_bug_reports[:n_train],
        all_bug_reports[n_train:n_train+n_val],
        all_bug_reports[n_train+n_val:]
    )

def prepare_dataset(data_folder_path, pickle_file, cache_file="processed_data.pt"):
    # Check if the cache of the pickle dataset exists - if yes, return the data from this cache data.
    cache_data_path = os.path.join(data_folder_path, cache_file)
    if os.path.exists(cache_data_path):
        logger.info(f"Loading processed data from cache: {cache_file}")
        cached_data = torch.load(cache_data_path, weights_only=False)
        return cached_data['train_data'], cached_data['val_data'], cached_data['test_data'], cached_data['vocab_size'], cached_data['bug_report_descriptions']
    
    # Extract all bug report information
    logger.info(f"Processing dataset from {pickle_file}")
    all_data_by_desc = defaultdict(list)
    bug_report_descriptions = {}
    unique_bug_reports = {}
    next_bug_report_id = 0
    all_bug_reports_texts = []
    
    try:
        with open(pickle_file, 'rb') as f:
            all_processed_data = pickle.load(f)
            if isinstance(all_processed_data, list):
                for instance in tqdm(all_processed_data, desc="Processing Data"):
                    if isinstance(instance, dict): # Ensure the element is a dictionary
                        bug_report_desc =  instance.get("bug_report")
                        filename = instance.get("filename")
                        ast_dict = instance.get("ast_src_code")
                        label = instance.get("label")

                        if isinstance(ast_dict, dict):  # Ensure AST is a dict
                            if bug_report_desc not in unique_bug_reports:
                                unique_bug_reports[bug_report_desc] = next_bug_report_id
                                bug_report_descriptions[next_bug_report_id] = bug_report_desc
                                next_bug_report_id += 1
                                all_bug_reports_texts.append(bug_report_desc)
                    
                            br_id = unique_bug_reports[bug_report_desc]
                            data = ast_to_pyg_graph(ast_dict)
                            data.y = torch.tensor([label], dtype=torch.long)
                            data.bug_report_id =  br_id
                            data.filename = filename
                            all_data_by_desc[bug_report_desc].append(data)
                        else:
                            logger.warning(f"AST is not a dictionary for instance: {filename}, type: {type(ast_dict)}")
                    else:
                        logger.warning(f"Unexpected data type in processed data (expecting dict): {type(instance)}, value: {instance}")
        
            else:
                logger.warning(f"Unexpected data type loaded from pickle (expecting list): {type(all_processed_data)}")
                return None, None, None, 0, None
    
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {pickle_file}")
        return None, None, None, 0, None
    except Exception as e:
        logger.error(f"Error Processing pickle file: {e}")
        return None, None, None, 0, None
    
    all_bug_report_ids = list(range(next_bug_report_id))
    train_ids, val_ids, test_ids = split_bug_reports(all_bug_report_ids)

    train_data = {br_id: [] for br_id in train_ids}
    val_data = {br_id: [] for br_id in val_ids}
    test_data = {br_id: [] for br_id in test_ids}

    for desc, data_list in all_data_by_desc.items():
        br_id = unique_bug_reports[desc]
        if br_id in train_ids:
            train_data[br_id].extend(data_list)
        elif br_id in val_ids:
            val_data[br_id].extend(data_list)
        elif br_id in test_ids:
            test_data[br_id].extend(data_list)

    # Create vocabulary from all bug report texts
    all_tokens = [token for text in all_bug_reports_texts for token in text.split()]
    vocab = {token: i for i, token in enumerate(sorted(list(set(all_tokens))))}
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size of bug reports: {vocab_size}")

    num_node_features = 1
    logger.info(f"Number of node features: {num_node_features}")

    processed_data_to_save = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'vocab_size': vocab_size,
        'bug_report_descriptions': bug_report_descriptions
    }

    torch.save(processed_data_to_save, cache_data_path)
    logger.info(f"Processed data saved to cache: {cache_file}")

    return train_data, val_data, test_data, vocab_size, bug_report_descriptions

            
   