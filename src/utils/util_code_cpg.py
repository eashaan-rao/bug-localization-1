'''
Code to extract Code Property Graphs (CPG) using joern tool
'''

import os
import subprocess
import pandas as pd
from tqdm import tqdm
import uuid
import shutil
import json
import pickle

JOERN_BIN = "/usr/local/bin/"

def create_temp_java_file(src_code, tmp_dir):
    uid = str(uuid.uuid4())
    filepath = os.path.join(tmp_dir, f"{uid}.java")
    with open(filepath, "w") as f:
        f.write(src_code)
    return filepath

def extract_cpg_with_joern(java_file, export_output_dir):
    try:
        # 1. Run Joern parse to Create CPG bin file
        subprocess.run([
            os.path.join(JOERN_BIN, "joern-parse"), 
            java_file, 
            "--language", "java"], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 3. Run joern with the script
        subprocess.run([
            os.path.join(JOERN_BIN, "joern-export"), 
            "--repr=all",
            "--format=graphson",
            f"--out={export_output_dir}"], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # 4. Read the generated .jsonl file
        json_path = os.path.join(export_output_dir, "export.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                if os.path.exists(export_output_dir):
                    shutil.rmtree(export_output_dir)
                return json.load(f)
        else:
            raise FileNotFoundError(f"Expected export JSON not found: {json_path}")
    except Exception as e:
        print(f"Failed to process {java_file}: {e}")
        return None
    
def save_batch_to_pickle(batch_data, output_pickle_path):
    with open(output_pickle_path, "ab") as f:
        pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def generate_cpg_dataset(csv_path, output_pickle_path, batch_size=1000):
    tmp_dir="temp_joern"
    os.makedirs(tmp_dir, exist_ok=True)
    batch=0

    for chunk in pd.read_csv(csv_path, chunksize=batch_size):
        batch_output = []

        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing Batch {batch + 1}"):
            src_code = row['source_code']
            bug_report = row['bug_report']
            filename = row['file name']
            label = row['label']

            java_file = create_temp_java_file(src_code, tmp_dir)
            output_dir = "outdir"

            cpg_json = extract_cpg_with_joern(java_file, output_dir)

            if cpg_json:
                batch_output.append({
                    "bug_report": bug_report,
                    "filename": filename,
                    "cpg": cpg_json,
                    "label": label
                })
            
            # Clean up per-file
            if os.path.exists(java_file):
                os.remove(java_file)
            if os.path.exists("cpg.bin"):
                os.remove("cpg.bin")
            
        
        # save each batch after processing
        if batch_output:
            save_batch_to_pickle(batch_output, output_pickle_path)
        batch += 1 

    # # Save results
    # with open(output_jsonl_path, "w") as f:
    #     for item in output_data:
    #         f.write(json.dumps(item) + "\n")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Saved full CPG dataset to {output_pickle_path}")

def extract_src_to_cpg():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    dataset_filepath = os.path.join(data_folder_path, 'dataset_codebert.csv')
    # output_json_file = os.path.join(data_folder_path, 'cpg_dataset.jsonl')
    output_pickle_file = os.path.join(data_folder_path, 'cpg_dataset.pkl')
    generate_cpg_dataset(dataset_filepath, output_pickle_file, batch_size=1000)