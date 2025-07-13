# Online dataset link: https://zenodo.org/records/268486 (bugfiles by Xin Ye)

from models.dnn_model import dnn_model_kfold
from models.rvsm_model import rvsm_model
from models.lambdamart import lambdaMART
from models.dnn_model_custom_loss import train_dnn_with_custom_loss
from models.gnn_ast_model import gcn_model

from feature_extraction import extract_features
from feature_extraction_codebert import prepare_dataset_for_codebert
from models.codeBERT_finetune import codebert_finetune
from utils.util_code_ast import extract_src_to_ast
from utils.util_code_cpg import extract_src_to_cpg

# Step 1: Extract Features from the Eclipse_Platform_UI.txt
# extract_features("aspectj")

# Step 2: rvsm model
# print("-"*60)
# print("RVSM Model")
# print(rvsm_model("aspectj"))
# print("-"*60)

# # Step 3: DNN-MLP Model
# print("-"*60)
# print("DNN-MLP Model")
# print(dnn_model_kfold("aspectj"))
# print("-"*60)

# # Step 4: LambdaMART Model
# print("-"*60)
# print("LambdaMART")
# print(lambdaMART("aspectj"))
# print("-"*60)

# # Step 5: DNN model with custom loss
# print("-"*60)
# print("DNN with custom loss Model")
# print(train_dnn_with_custom_loss("aspectj"))
# print("-"*60)

# Step 6: Extract dataset for fine tuning code bert
# prepare_dataset_for_codebert("eclipse_platform_ui")

# # Step 7: Finetune the codebert
codebert_finetune("eclipse_platform_ui")

# # Step 8: Convert src code to ast
# extract_src_to_ast("aspectj")

# # Step 9: Convert src code to cpg
# extract_src_to_cpg()

# # Step 10: NN based ast representation bug localization
# gcn_model("jdt")



