# Online dataset link: https://zenodo.org/records/268486 (bugfiles by Xin Ye)

from models.dnn_model import dnn_model_kfold
from models.rvsm_model import rvsm_model
from models.lambdamart import lambdaMART
from models.dnn_model_custom_loss import train_dnn_with_custom_loss

from feature_extraction import extract_features
from feature_extraction_codebert import prepare_dataset_for_codebert
from models.codeBERT_finetune import codebert_finetune

# Step 1: Extract Features from the Eclipse_Platform_UI.txt
# extract_features()
# Extract dataset for fine tuning code bert
# prepare_dataset_for_codebert()

# Finetune the codebert
codebert_finetune()

# BLs model
# 1. rvsm model
# print(rvsm_model())

# 2. dnn model
# print(dnn_model_kfold(10))

# 3. LambdaMART
# print(lambdaMART())

# 4. DNN model with custom loss
# print(train_dnn_with_custom_loss())


