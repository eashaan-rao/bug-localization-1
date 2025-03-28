'''
Helper functions to support transformer-based bug localization model.
using a pre-trained codeBERT model.
'''


from transformers import AutoTokenizer, AutoModel
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Load CodeBERT
model = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Move model to GPU
model.to(device)

def split_chunks(text, chunk_size=512, overlap=50):
    '''
    Splits the source code into overlapping chunks.

    Args:
        src (str) : Source code
        chunk_size (int): Size of each chunk
        overlap (int): Overlapping tokens between chunks

    Returns: 
        list: list of chunks.
    '''
    # tokens = tokenizer.tokenize(text)
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i: i + chunk_size]
        if chunk:  # Ensure chunk is not empty
            chunks.append(tokenizer.convert_tokens_to_string(chunk))

    return chunks

def truncate_br_text(br_text, max_tokens=512):
    '''
    Dynamically truncate br_text based on available space after src_chunk.

    Args:
        br_text (str) : bug report text
        src_chunk (str):  Current chunk of source code
        max_tokens (int): Max allowed tokens (Default 512 for codeBERT)

    Returns:
        str: Truncated bug report text.
    '''

    # Tokenize br_text and truncate if needed
    br_tokens = tokenizer.tokenize(br_text)
    if len(br_tokens) > max_tokens:
        br_tokens = br_tokens[:max_tokens]

    return tokenizer.convert_tokens_to_string(br_tokens)


def codebert_similarity(br_text, src):
    '''
    Compute similarity score between bug report text and source code using CodeBERT.

    Args:
        br_text (str): Bug report text.
        src (str): source code content

    Returns: 
        float: Cosing similarity between embeddings.
    '''

    try:
        src_chunks = split_chunks(src, chunk_size=512, overlap=50)
        if not src_chunks:
                print("No chunks generated. Returning 0.0 similarity")
                return 0.0
        
        similarities = []

        # Truncate and get embeddings for br_text separately
        br_text_processed = truncate_br_text(br_text)

        # Tokenize and get embeddings for br_text separately
        br_inputs = tokenizer(
            br_text_processed,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            br_outputs = model(**br_inputs)
            br_embedding = br_outputs.last_hidden_state.mean(dim=1)  # Shape: [1, 768]

        for chunk in src_chunks:
            if not chunk.strip():
                print("Empty chunk encountered. Skipping...")
                continue # Skip empty chunks
            # Tokenize and get embeddings for each src_chunk
            src_inputs = tokenizer(
                chunk,
                return_tensors="pt", 
                padding="max_length", 
                truncation=True,  
                max_length=512
            ).to(device)

            # Generate src chunk embeddings
            with torch.no_grad():
                src_outputs = model(**src_inputs)
                src_embedding = src_outputs.last_hidden_state.mean(dim=1)  # shape: [1, 768]

            # print(f"Inputs shape: {inputs['input_ids'].shape}")
            # print(f"Embeddings shape: {embeddings.shape}")

            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(br_embedding, src_embedding, dim=1).item()
            similarities.append(similarity)
            
        return max(similarities) if similarities else 0.0
    
    except Exception as e:
        print(f"Error in CodeBERT similarity: {e}")
        return 0.0
    