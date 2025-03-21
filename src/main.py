# Online dataset link: https://zenodo.org/records/268486 (bugfiles by Xin Ye)

from models.dnn_model import dnn_model_kfold
from models.rvsm_model import rvsm_model
from models.lambdamart import lambdaMART
from feature_extraction import extract_features
# Step 1: Extract Features from the Eclipse_Platform_UI.txt
# extract_features()

# BLs model
# 1. rvsm model
# print(rvsm_model())

# 2. dnn model
# print(dnn_model_kfold(10))

# 3. LambdaMART
# print(lambdaMART())


