from tree_sitter import Language, Parser
import tree_sitter_java as tsjava
from tree_sitter_languages import get_language, get_parser
import json
import os
import pickle
import pandas as pd
from tqdm import tqdm


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
    # Nodes to be completely excluded
    EXCLUDED_NODE_TYPES = {".", ",", "{", "}", "(", ")", "*", "package", "import", 
                           "block_comment", "line_comment", "asterisk", "="}
    
    # Nodes whose type is kept but text is removed
    INCLUDE_NODE_EXCLUDE_TEXT = {"package_declaration", "import_declaration", "scoped_identifier", 
                                 "class_declaration", "public", "class", "implements", "class_body", 
                                 "private", "field_declaration", "variable_declaration", 
                                 "constructor_declaration", "formal_parameters", "constructor_body", 
                                 "if_statement", "throw_statement", "object_create_expression", 
                                 "expression_statement", "method_declaration", "block", 
                                 "local_variable_declaration", "interface_body", "interface_declaration",
                                "for_statement", "try_statement", "argument_list", "method_invocation"}
    
    #  Skip the "program" node altogether as it is providing whole src code as text.
    if node.type == "program":
        return [ast_to_dict(child, source_code) for child in node.children]
    
    if node.type in EXCLUDED_NODE_TYPES:
        return None # Completely remove these nodes
    
    ast_dict = {"type": node.type}
    if node.type not in INCLUDE_NODE_EXCLUDE_TEXT:
        ast_dict["text"] = source_code[node.start_byte:node.end_byte].strip()

    children = [ast_to_dict(child, source_code) for child in node.children]
    children = [child for child in children if child is not None] # Remove None entries

    # Recursively process children
    if node.children:
        ast_dict["children"] = children
    return ast_dict


def process_csv_to_ast(input_csv, output_pickle):
    '''
    Process Java source codes into ASTs in chunks and store in Pickle format
    '''

    results = []
    batch = 0
    chunk_size = 1000
    
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        batch_results = []
        for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Processing Batch {batch + 1}"):
            bug_report = row['bug_report']
            filename = row['file name']
            src_code = row['source_code']
            label = row['label']

            ast_representation = parse_java_code(src_code)
            batch_results.append({
                "bug_report": bug_report,
                "filename": filename,
                "ast_src_code": ast_representation,
                "label": label
            })
        
        # # Save as JSON
        # with open(output_json, "w", encoding="utf-8") as f:
        #     json.dump(results, f, indent=2)
        # print(f"AST dataset saved to {output_json}")


        # Save each batch incrementally 
        with open(output_pickle, 'ab') as pickle_file:
            pickle.dump(batch_results, pickle_file)

        print(f"Processed batch {batch + 1}, saved {len(batch_results)} records to {output_pickle}")
        batch += 1

    print(f"AST dataset saved to {output_pickle}")

def extract_src_to_ast():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    data_folder_path = os.path.join(parent_dir, 'data')
    dataset_filepath = os.path.join(data_folder_path, 'dataset_codebert.csv')
    # output_json_file = os.path.join(data_folder_path, 'ast_dataset.json')
    output_pickle_file = os.path.join(data_folder_path, 'pickle_dataset.pkl')
    process_csv_to_ast(dataset_filepath, output_pickle_file)