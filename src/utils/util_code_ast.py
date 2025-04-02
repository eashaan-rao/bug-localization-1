from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
from tree_sitter_languages import get_language, get_parser
import json
import os
import pandas as pd


# JAVA_LANGUAGE = '/home/user/CS21D002_A_Eashaan_Rao/Research/Bug_Localization_Replication/tree-sitter-java'
# Language.build_library(
#     'build/my-language.so',
#     [JAVA_LANGUAGE]
# )

# java_language = get_language('java')
# parser = get_parser(java_language)

# JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
JAVA_LANGUAGE = Language(tsjava.language())
parser = Parser(JAVA_LANGUAGE)
# parser.set_language(JAVA_LANGUAGE)

def parse_java_code(source_code):
    '''
    Parse Java code and return AST as a dictionary.
    '''
    tree = parser.parse(bytes(source_code, 'utf-8'))
    root_node = tree.root_node
    return ast_to_dict(root_node, source_code)

def ast_to_dict(node, source_code):
    '''
    Convert Tree-sitter AST to a dictionary format.
    '''
    #  Skip the "program" node altogether as it is providing whole src code as text.
    if node.type == "program":
        return [ast_to_dict(child, source_code) for child in node.children]
    
    ast_dict = {
        "type": node.type,
        "text": source_code[node.start_byte:node.end_byte].strip()
    }
         
    # Recursively process children
    if node.children:
        ast_dict["children"] = [ast_to_dict(child, source_code) for child in node.children]
    return ast_dict


def process_csv_to_ast(input_csv, output_json, num_rows=2):
    '''
    Read a CSV file, process the first 'num_rows' Java source codes into ASTs, and store into JSON.
    '''
    df = pd.read_csv(input_csv)
    results = []
    for _, row in df.head(num_rows).iterrows():
        bug_report = row['bug_report']
        filename = row['file name']
        src_code = row['source_code']
        label = row['label']

        ast_representation = parse_java_code(src_code)
        results.append({
            "bug_report": bug_report,
            "filename": filename,
            "ast_src_code": ast_representation,
            "label": label
        })
        

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"AST dataset saved to {output_json}")

def extract_src_to_ast():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    dataset_filepath = os.path.join(data_folder_path, 'dataset_codebert.csv')
    output_json_file = os.path.join(data_folder_path, 'ast_dataset.json')
    process_csv_to_ast(dataset_filepath, output_json_file)