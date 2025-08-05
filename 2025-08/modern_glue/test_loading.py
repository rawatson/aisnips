import os
# 1. Tell the library to work in offline mode
os.environ['HF_DATASETS_OFFLINE'] = "1"
from datasets import load_dataset

print("Loading CoLA validation dataset...")
dataset = load_dataset('glue', 'cola', split='validation')
print("Load complete!")

# For posterity, the HF_DATASETS_OFFLINE var needs to be set BEFORE you import the datasets library.
# When offline, I can load COLA in 0.8s
# When online, it takes 4.3s to check for a more recent version of the dataset (which there obviously isn't)