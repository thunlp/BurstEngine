from datasets import load_dataset
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
train_data = load_dataset('bookcorpus/bookcorpus', split='train')
train_data.to_json("BookCorpusDataset_text_document.json", lines=True)
