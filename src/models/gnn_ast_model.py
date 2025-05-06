'''
Code for training a GNN model for AST based source code for bug localization
'''

import torch
from utils.util_gnn_ast_data import prepare_dataset
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout
from torch.utils.data import Dataset
from collections import defaultdict
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import logging
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Initilaize tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Set up logging
logging.basicConfig(
    level = logging.INFO,
    format = '%s(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class BugLocalizationModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_node_features, hidden_channels, num_gnn_layers=3,
                 gnn_dropout_rate=0.1, use_gat=False, num_attention_heads=1, dropout_rate=0.1):
        super(BugLocalizationModel, self).__init__()
        # Bug report text encoder
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.text_encode = torch.nn.LSTM(embedding_dim, hidden_channels, batch_first=True)
        self.text_norm = LayerNorm(hidden_channels)
        self.text_dropout = Dropout(dropout_rate)

        # Use GCN only for code representation
        self.ast_encoder = GCN(num_node_features, hidden_channels, num_layers=num_gnn_layers, 
                               dropout_rate=gnn_dropout_rate, use_gat=use_gat, num_attention_heads=num_attention_heads)

        # Combination and ranking components
        combination_input_dim = hidden_channels + (hidden_channels * num_attention_heads if use_gat else hidden_channels)
        self.combination_layer = torch.nn.Linear(combination_input_dim, hidden_channels)
        self.comb_norm = LayerNorm(hidden_channels)
        self.comb_dropout = Dropout(dropout_rate)
        self.ranking_score = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, bug_report_tokens, ast_data):
        # 1. encode bug report
        embedded = self.embedding(bug_report_tokens)
        _, (hidden, _) = self.text_encode(embedded)
        # Handle batch dimension properly - hidden shape is (bnum_layers, batch, hidden_size)
        bug_repr = hidden[-1] # Take the last layer's hidden state
        bug_repr = self.text_norm(bug_repr)
        bug_repr = self.text_dropout(bug_repr)

        # 2. Encode AST using existing GCN
        # The ast_encoder expects the entire Pytorch geometri data object
        code_repr = self.ast_encoder(ast_data.x.float(), ast_data.edge_index, ast_data.batch)

        # 3. Combine representations
        combined = torch.cat([bug_repr, code_repr], dim=1)
        combined = F.relu(self.combination_layer(combined))
        combined = self.comb_norm(combined)
        combined = self.comb_dropout(combined)

        # 4. Calculate ranking score
        score = self.ranking_score(combined)
        return score

# GCN model (Graph Classification) 
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers=3, dropout_rate=0.1, use_gat=False, num_attention_heads=1):
        """
        Deeper Graph Convolution Network with option for GAT (Graph Attention layers) for Bug Localization.

        Args:
            um_node_features (int): Dimension of node features.
            hidden_channels (int): Dimension of hidden layers.
            num_layers (int): Number of GCN/GAT layers. Default is 3.
            dropout_rate (float): Dropout probability. Default is 0.1.
            use_gat (bool): Whether to use GATConv layers instead of GCNConv. Default is False.
            num_attention_heads (int): Number of attention heads if using GAT. Default is 1.
        """
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = Dropout(dropout_rate)
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.use_gat = use_gat
        self.num_attention_heads = num_attention_heads

        if use_gat:
            self.convs.append(GATConv(num_node_features, hidden_channels, heads=num_attention_heads))
        else:
            self.convs.append(GCNConv(num_node_features, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels * num_attention_heads if use_gat else hidden_channels))

        for _ in range(num_layers - 1):
            if use_gat:
                self.convs.append(GATConv(hidden_channels * num_attention_heads, hidden_channels, heads=num_attention_heads))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels * num_attention_heads if use_gat else hidden_channels))
            
        # self.conv1 = GCNConv(num_node_features, hidden_channels)
        # self.norm1 = LayerNorm(hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.norm2 = LayerNorm(hidden_channels)
        # self.dropout = Dropout(dropout_rate)
        # No classification layer needed

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(self.norms[i](x))
            x = self.dropout(x)
        

        # # First graph conv layer
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.dropout(x)

        # # Second graph conv layer
        # x = self.conv2(x, edge_index)
        # x = F.relu(self.norm2(x))

        # global mean pooling to get graph-level representation
        x = global_mean_pool(x, batch)

        return x  # return the graph embeddings directly

# ----- Helper Functions -----

def get_bug_report_tokens(bug_report_id, all_bug_report_descriptions, tokenizer, device):
    '''
    Retrieves, tokenizes, and converts a bug report description to a torch.Tensor

    Args:
        bug_report_id (str) : The description of the bug report.
        all_bug_report_descriptions (dict) : Dictionary mapping bug report IDs to their descriptions.
        tokenizer: Tokenizer object (e.g., from transformers library)
        device: Device to move the tensor to

    Returns:
        torch.Tensor: Tokenize bug report description. 
    '''
    description = all_bug_report_descriptions.get(bug_report_id,"")
    tokens = tokenizer(description, return_tensors="pt", truncation=True, padding=True)
    return tokens['input_ids'].squeeze(0).to(device)

def pairwise_ranking_loss(scores, labels, margin=1.0):
    '''
    Pairwise ranking loss.

    Args:
        scores (torch.Tensor): Predicted scores for the files.
        labels (torch.Tensor): Binary labels (1 for buggy, 0 for non-buggy).
        margin (float): Margin for the loss.

    Returns:
        torch.Tensor: The pairwise ranking loss
    '''
    loss = torch.tensor(0.0, requires_grad=True, device=scores.device)
    positive_indices = (labels == 1).nonzero(as_tuple=True)[0]
    negative_indices = (labels == 0).nonzero(as_tuple=True)[0]

    if len(positive_indices) > 0 and len(negative_indices) > 0:
        for pos_idx in positive_indices:
            for neg_idx in negative_indices:
                diff = scores[neg_idx] - scores[pos_idx] + margin
                loss = loss + torch.max(torch.zeros_like(diff), diff)
        loss = loss/ (len(positive_indices) * len(negative_indices) + 1e-7)
    return loss

def predict_and_rank(model, data_by_bug_report, all_bug_report_description, tokenizer, device):
    '''
    Generate predictions for ranks files for each bug report

    Args:
        model (BugLocalization): Trained model
        data_by_bug_report (dict): Dictionary mapping bug report IDs to a list of Data objects.
        all_bug_report_descriptions (dict): Dictionary of bug report descriptions.
        tokenizer: Tokenizer object
        device: Device to run inference on

    Returns:
        dict: Bug report ID to list of (filename, label, score) tuples, sorted by score
    '''
    model.eval()
    bug_report_to_ranked_results = defaultdict(list)

    with torch.no_grad():
        for bug_report_id, files in tqdm(data_by_bug_report.items(), desc ="Predicting and Ranking"):
            if not files:
                continue
            bug_report_tokens = get_bug_report_tokens(bug_report_id, all_bug_report_description, tokenizer, device).unsqueeze(0).repeat(len(files), 1)
            ast_data_list = [f.to(device) for f in files]
            batch_graph = Batch.from_data_list(ast_data_list)
            scores = model(bug_report_tokens, batch_graph).squeeze().cpu().numpy()
            filenames = [f.filename for f in files]
            labels = [f.y.item() for f in files]
            
            results = []
            for filename, label, score in zip(filenames, labels, scores):
                results.append({'filename': filename, 'label':label, 'score': score})

            # Sort results by score in descending order
            bug_report_to_ranked_results[bug_report_id] = sorted(results, key=lambda x: x['score'], reverse=True)

    return bug_report_to_ranked_results


def compute_ranking_metrics(bug_report_to_ranked_results, k_values=[1, 5, 10]):
    '''
    Compute ranking metrics (top-k, MAP, MRR)

    Args:
        bug_report_to_ranked_results (dict): Output of predict_and_rank()
        k_values (list): List of Top-k values 

    Returns:
        tuple: (top-k accuracy dict, MAP score, MRR score)
    '''
    topk_accuracies = {k: [] for k in k_values}
    ap_scores = []
    rr_scores = []

    for bug_report, results in bug_report_to_ranked_results.items():
        ranked_labels = [item['label'] for item in results]
        scores = [item['score'] for item in results]

        # top-k accuracy
        for k in k_values:
            if k <= len(ranked_labels):
                topk = ranked_labels[:k]
                topk_accuracies[k].append(1 if 1 in topk else 0)  # 1 if at least one buggy file in top-k

        # MAP (mean average precision)
        if sum(ranked_labels) > 0: # only if there are buggy files
            ap_scores.append(average_precision_score(ranked_labels, scores))

        # MRR (mean reciprocal rank)
        try:
            first_hit = ranked_labels.index(1)
            rr_scores.append(1.0 / (first_hit + 1))
        except ValueError:
            rr_scores.append(0.0)

    # Calculate final metrics
    topk = {k: np.mean(topk_accuracies[k]) if topk_accuracies[k] else 0 for k in k_values}
    map_score = np.mean(ap_scores) if ap_scores else 0
    mrr_score = np.mean(rr_scores) if rr_scores else 0

    return topk, map_score, mrr_score

def visualize_metrics(train_losses, val_mrrs, val_maps, val_top1s, val_top5s, val_top10s, epochs):
    """
    Visualizes the training loss and validation metrics over epochs.

    Args:
        train_losses (list): List of training losses per epoch.
        val_mrrs (list): List of validation MRR per epoch.
        val_maps (list): List of validation MAP per epoch.
        val_top1s (list): List of validation Top@1 accuracy per epoch.
        val_top5s (list): List of validation Top@5 accuracy per epoch.
        val_top10s (list): List of validation Top@10 accuracy per epoch.
        epochs (int): Total number of epochs.
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(epochs_range, val_mrrs, label='Val MRR', color='orange')
    plt.title('Validation MRR')
    plt.xlabel('Epoch')
    plt.ylabel('MRR')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(epochs_range, val_maps, label='Val MAP', color='green')
    plt.title('Validation MAP')
    plt.xlabel('Epoch')
    plt.ylabel('MAP')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(epochs_range, val_top1s, label='Val Top@1', color='red')
    plt.title('Validation Top@1 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(epochs_range, val_top5s, label='Val Top@5', color='purple')
    plt.title('Validation Top@5 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(epochs_range, val_top10s, label='Val Top@10', color='brown')
    plt.title('Validation Top@10 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('gcn_model.png')  # Or choose your desired filename and format


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
    pickle_file = os.path.join(data_folder_path, 'ast_dataset.pkl')

    # Hyperparameters
    hidden_channels = 64
    embedding_dim = 128 
    learning_rate = 0.001
    weight_decay = 5e-4
    file_pair_batch_size=16  # Define the batch size for file pairs
    epochs = 20
    num_gnn_layers = 4
    gnn_dropout_rate = 0.2
    use_gat = True
    num_attention_heads = 2
    dropout_rate = 0.1

    train_dataset, val_dataset, test_dataset, vocab_size, bug_report_descriptions = prepare_dataset(data_folder_path, pickle_file)

    if not all([train_dataset, val_dataset, test_dataset, vocab_size, bug_report_descriptions]):
        logger.error("Failed to load dataset.")
        return 
    
    train_bug_ids = list(train_dataset.keys())
    val_bug_ids = list(val_dataset.keys())
    test_bug_ids = list(test_dataset.keys())

    model = BugLocalizationModel(vocab_size, embedding_dim, 1, hidden_channels, num_gnn_layers, gnn_dropout_rate,
                                 use_gat, num_attention_heads, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ----- Training Loop (Bug Report Centric) -----
    logger.info("Starting training with bug report centric processing...")
    # Lists to store metrics for visualization
    train_losses = []
    val_mrrs = []
    val_maps = []
    val_top1s = []
    val_top5s = []
    val_top10s = []

    best_val_mrr = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        num_bug_reports_processed = 0

        for bug_report_id in tqdm(train_bug_ids, desc=f"Epoch {epoch}"):
            bug_report_files = train_dataset.get(bug_report_id, [])
            if not bug_report_files:
                continue

            # Create a function to get tokenizes bug report for an ID
            bug_report_tokens = get_bug_report_tokens(bug_report_id, bug_report_descriptions, tokenizer, device).unsqueeze(0)

            num_files = len(bug_report_files)
            for i in range(0, num_files, file_pair_batch_size):
                batch_files = bug_report_files[i: i + file_pair_batch_size]
                batch_ast_data = [file.to(device) for file in batch_files]

                # Repeat bug report tokens for the batch size
                batch_bug_report_tokens = bug_report_tokens.repeat(len(batch_files), 1)

                # Create a PyTorch Geometric Batch object for the ASTs
                batch_graph = Batch.from_data_list(batch_ast_data)
                
                # Forward pass
                scores = model(batch_bug_report_tokens, batch_graph).squeeze()
                labels = torch.cat([file.y for file in batch_files]).float().to(device)

                # Calculate ranking loss 
                loss = pairwise_ranking_loss(scores, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_files)
            
            num_bug_reports_processed += 1 

        avg_loss = total_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
        logger.info(f"Epoch: {epoch:03d}, Train Loss: {avg_loss:.4f}")
        train_losses.append(avg_loss)

        # --- Validation ----
        model.eval()
        val_ranked_results = predict_and_rank(model, val_dataset, bug_report_descriptions, tokenizer, device)
        topk_val, map_val, mrr_val = compute_ranking_metrics(val_ranked_results)
        logger.info(f"Epoch: {epoch:03d}, Val MRR: {mrr_val:.4f}, Val MAP: {map_val:.4f}, Val Top@1: {topk_val.get(1,0):.4f}")
        val_mrrs.append(mrr_val)
        val_maps.append(map_val)
        val_top1s.append(topk_val.get(1, 0))
        val_top5s.append(topk_val.get(5, 0))
        val_top10s.append(topk_val.get(10, 0))

        if mrr_val > best_val_mrr:
            best_val_mrr = mrr_val
            best_model_state = model.state_dict().copy()

    # ---- Testing ----
    if best_model_state:
        model.load_state_dict(best_model_state)
    model.eval()
    test_ranked_results = predict_and_rank(model, test_dataset, bug_report_descriptions, tokenizer, device)
    topk_test, map_test, mrr_test = compute_ranking_metrics(test_ranked_results)
    logger.info("Test Bug Localization Performance:")
    logger.info(f"MRR: {mrr_test: .4f}, MAP: {map_test: .4f}")
    for k, acc in topk_test.items():
        logger.info(f"Top-{k} Accuracy: {acc:.4f}")

    visualize_metrics(train_losses, val_mrrs, val_maps, val_top1s, val_top5s, val_top10s, epochs)

    # Save model
    output_dir = os.path.join(data_folder_path, "models")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'bug_localization_gnn.pt'))
    logger.info(f"Model saved to {os.path.join(output_dir, "bug_localization_gnn.pt")}")


    