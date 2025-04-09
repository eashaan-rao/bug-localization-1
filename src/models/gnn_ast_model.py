'''
Code for training a GNN model for AST based source code for bug localization
'''

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
import random
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import logging

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%s(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# GCN model (Graph Classification) 
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=2):
        """
        Graph Convolution Network for Bug Localization.

        Args:
            num_node_featires (int): Dimension of node features
            hidden_channels (int): Dimension of hidden layers
            num_classes (int): Number of output classes (default: 2 for buggy vs non_buggy)
        """
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)  
    
    def forward(self, data):
        """Forward pass through the network"""
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        # First graph conv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Second graph conv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Classification layer
        x = self.lin(x)
        return x

# Custom InMemoryDataset for AST graphs
class ASTBugReportDataset(InMemoryDataset):
    def __init__(self, root, pickle_file, split_bug_ids=None, transform=None, pre_transform=None):
        '''
        Dataset for AST Bug Localization

        Args:
            root(str) : Root directory where the processed data will be stored.
            pickle_file(str): Path of the pickle file containing the AST dataset
            split_bug_ids (list, optional): Bug IDs to include in this dataset split
            transform (callable, optional): Transform to be applied to each data instance
            pre_transform (callable, optional): Pre-transform to be applied to each data instance
        '''

        self.pickle_file = pickle_file
        self.split_bug_ids = split_bug_ids
        super(ASTBugReportDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            # not used because we assume the raw file is already in pickle format...
            return []
        
        @property
        def processed_file_names(self):
            return ["data.pt"]
        
        def process(self):
            # Process the raw data into PyTorch Geometric Data Objects
            logger.info(f"Processing dataset from {self.pickle_file}")
            
            # check if file exists
            if not os.path.exists(self.pickle_file):
                raise FileNotFoundError(f"Dataset file not found: {self.pickle_file}")
            
            # Load bug reports based on split
            data_list = []
            samples = defaultdict(list)
    
            with open(self.pickle_file, 'rb') as f:
                try:
                    # First, detemine the format of the pickle file by looking at the first entry
                    first_entry = pickle.load(f)
                    f.seek(0)  # Reset file pointer

                    # Process the file based on its format
                    if isinstance(first_entry, dict):
                        # Format: List of dictionaries
                        dataset = pickle.load(f)
                        for instance in tqdm(dataset, desc="Converting ASTs to Graphs"):
                            # Each instance is expected to have keys: "bug_report", "filename", "ast_src_code," and "label"
                            br_id = instance.get("bug_report", "unknown")

                            # Skip if not in split_bug_ids (if provided)
                            if self.split_bug_ids and br_id not in self.split_bug_ids:
                                continue

                            ast_dict = instance["ast_src_code"]
                            label = instance["label"]
                            filename = instance.get("filename", "unknown")

                            # Convert the AST to a PyG graph
                            data = ast_to_pyg_graph(ast_dict)
                            data.y = torch.tensor([label], dtype=torch.long)
                            data.bug_report_id = br_id
                            data.filename = filename
                            data_list.append(data)
                    else:
                        # Format: Sequential entries
                        while True:
                            try:
                                br_id, filename, ast_dict, label = pickle.load(f)

                                # Skip if not in split_bug_ids (if provided)
                                if self.split_bug_ids and br_id not in self.split_bug_ids:
                                    continue

                                 # Convert the AST to a PyG graph
                                data = ast_to_pyg_graph(ast_dict)
                                data.y = torch.tensor([label], dtype=torch.long)
                                data.bug_report_id = br_id
                                data.filename = filename
                                data_list.append(data)
                            except EOFError:
                                break
                except Exception as e:
                    logger.error(f"Error processing pickle file: {e}")
                    raise
            
            if not data_list:
                logger.warning("No data instances were processed. Dataset maybe empty or invalid.")
            
            logger.info(f"Proceesed {len(data_list)} data instances")

            # Collate all data objects
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

 
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

        # Add edge from parent (if not roor)
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
    bug_reports = list(set(all_bug_reports))
    random.shuffle(bug_reports)
    
    n = len(bug_reports)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    return (
        bug_reports[:n_train],
        bug_reports[n_train:n_train+n_val],
        bug_reports[n_train+n_val:]
    )


def train_model(model, loader, optimizer, device):
    '''
    Train the model for one epoch

    Args:
        model (GCN): Model to train
        loader (DataLoader) : Training data loader
        optimizer: Optimizer for training
        device: Device to train on

    Returns:
        float: Average loss for this epoch
    '''
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += int((pred == data.y.view(-1)).sum())
        total += data.y.size(0)


        total_loss += loss.item() * data.num_graphs

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy
    

def evaluate(model, loader, device):
    '''
    Evaluate model on validation or test set

    Args:
        model (GCN): Model to evaluate
        loader (DataLoader): Data loader for evaluation
        device: Device to evaluate on
    
    Returns:
        tuple: (accuracy, predictions, targets)
    '''
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)

            # Store predictions and targets
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(data.y.view(-1).cpu().numpy())

            # Calculate accuracy
            correct += int((pred == data.y.view(-1).sum()))
            total += data.y.size(0)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, all_preds, all_targets


def predict_and_rank(model, dataloader, device):
    '''
    Generate predictions for ranking files by bug probability

    Args:
        model (GCN): Trained model
        dataloader (DataLoader): Test data loader
        device: Device to run inference on

    Returns:
        dict: Bug report ID to list of (filename, label, score) tuples
    '''
    model.eval()
    bug_report_to_scores = defaultdict(list)

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out = model(data)
            # Get probability of buggy class
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

            batch_size = data.y.size(0)
            for i in range(batch_size):
                # Extract single graph data
                bug_report = data.bug_report_id[i] 
                filename = data.filename[i]  
                label = int(data.y[i].item())
                score = float(probs[i])

                bug_report_to_scores[bug_report].append((filename, label, score))
        
    return bug_report_to_scores

def compute_ranking_metrics(bug_report_to_scores, k_values=[1, 5, 10]):
    '''
    Compute ranking metrics for bug localization

    Args:
        bug_report_to_scores (dict): Bug report ID to list of (filename, label, score) tuples
        k_values (list): Top-k values to compute accuracy for

    Returns:
        tuple: (top-k accuracy dict, MAP score, MRR score)
    '''
    topk_accuracies = {k: [] for k in k_values}
    ap_scores = []
    rr_scores = []

    for bug_report, results in bug_report_to_scores.items():
        # Sort by predicted bug score in descending order
        results.sort(key=lambda x: x[2], reverse=True) 
        filenames = [f for f, _, _ in results]
        labels = [label for _, label, _ in results]
        scores = [score for _, _, score in results]

        # top-k accuracy
        for k in k_values:
            if k <= len(labels):
                topk = labels[:k]
                topk_accuracies[k].append(1 if 1 in topk else 0)  # 1 if at least one buggy file in top-k

        # MAP (mean average precision)
        if sum(labels) > 0: # only if there are buggy files
            ap_scores.append(average_precision_score(labels, scores))

        # MRR (mean reciprocal rank)
        try:
            first_hit = labels.index(1)
            rr_scores.append(1.0 / (first_hit + 1))
        except ValueError:
            rr_scores.append(0.0)

    # Calculate final metrics
    topk = {k: np.mean(topk_accuracies[k]) for k in k_values}
    map_score = np.mean(ap_scores) if ap_scores else 0
    mrr_score = np.mean(rr_scores) if rr_scores else 0

    return topk, map_score, mrr_score



# Training and evaluation
def gcn_model():
    '''
    Main function to train and evaluate the bug localization model.
    '''
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Paths
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    pickle_file = os.path.join(data_folder_path, 'pickle_dataset.pkl')

    # Hyperparameters
    hidden_channels = 64
    learning_rate = 0.01
    weight_decay = 5e-4
    epochs = 50

    # Check if dataset exists
    if not os.path.exists(pickle_file):
        logger.error(f"Dataset file not found: {pickle_file}")
        return 
    
    # Extract all bug report IDs
    logger.info("Extracting bug report IDs from dataset...")
    all_bug_reports = set()
    try:
        with open(pickle_file, 'rb') as f:
            # Try to determine the format of the pickle file
            try:
                first_entry = pickle.load(f)
                f.seek(0)  # Reset file pointer

                if isinstance(first_entry, dict):
                    # Format: List of dictionaries
                    dataset = pickle.load(f)
                    for instance in dataset:
                        br_id = instance.get("bug_report", "unknown")
                        all_bug_reports.add(br_id)
                else:
                    # Format: Sequential entries
                    while True:
                        try:
                            br_id, _, _, _ = pickle.load(f)
                            all_bug_reports.add(br_id)
                        except EOFError:
                            break

            except Exception as e:
                logger.error(f"Error reading pickle file: {e}")
                return 
            
    except Exception as e:
        logger.error(f"Error opening pickle file: {e}")
        return
    
    logger.info(f"Found {len(all_bug_reports)} unique bug reports")

    # Instantiate dataset
    dataset = ASTBugReportDataset(root=data_folder_path, pickle_file=pickle_file)
    
    # loading ast pickle file to get all bug report IDs.
    with open(pickle_file, 'rb') as f:
        all_bug_reports = set()
        while True:
            try:
                br, _, _, _ = pickle.load(f)
                all_bug_reports.add(br)
            except EOFError:
                break
    
    # Dataloader setup
    train_ids, val_ids, test_ids = split_bug_reports(all_bug_reports)

    train_dataset = ASTBugReportDataset('pickle_dataset.pkl', train_ids)
    val_dataset = ASTBugReportDataset('pickle_dataset.pkl', val_ids)
    test_dataset = ASTBugReportDataset('pickle_dataset.pkl', test_ids)
    
    # each batch is grouped by one bug report (i.e., 50–52 files), we treat that as the “unit” of batching
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=bug_report_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=bug_report_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=bug_report_collate)

    # Determine number of node features (we used a single integer per node)
    num_node_features = 1 # since each node is represented by an integer index

    model = GCN(num_node_features, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), le=0.01, weight_decay=5e-4)

    



