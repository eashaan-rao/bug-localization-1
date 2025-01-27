# Online dataset link: https://zenodo.org/records/268486 (bugfiles by Xin Ye)

from dnn_model import dnn_model_kfold
from rvsm_model import rvsm_model
from feature_extraction import extract_features
# Step 1: Extract Features from the Eclipse_Platform_UI.txt

extract_features()

# print(dnn_model_kfold(10))
print(rvsm_model())