import os
from argparse import Namespace
import collections
import copy
import json
import numpy as np
import pandas as pd
import torch
from collections import Counter
import string
import modules.rnnlib  as rnnlib
from modules.rnnlib import Vocabulary, SequenceVocabulary, NewsVectorizer, NewsDataset, NewsModel, Inference, InferenceDataset
   
# Arguments
args = Namespace(
    seed=1234,
    cuda=True,
    shuffle=True,
    data_file='data/breaking-news.csv',
    vectorizer_file="breaking_news_vectorizer.json",
    model_state_file="breaking_news_model.pth",
    save_dir="models",
    train_size=0.90,
    val_size=0.10,
    test_size=0.15,
    pretrained_embeddings=None,
    cutoff=25, # token must appear at least <cutoff> times to be in SequenceVocabulary
    num_epochs=5,
    early_stopping_criteria=5,
    learning_rate=1e-3,
    batch_size=64,
    embedding_dim=100,
    rnn_hidden_dim=128,
    hidden_dim=100,
    num_layers=1,
    bidirectional=False,
    dropout_p=0.1,
)

# Set seeds
rnnlib.set_seeds(seed=args.seed, cuda=args.cuda)

# Create save dir
rnnlib.create_dirs(args.save_dir)

# Expand filepaths
args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# Raw data
df = pd.read_csv(args.data_file, header=0)
df.head()

# Split by category
by_category = collections.defaultdict(list)
for _, row in df.iterrows():
    by_category[row.category].append(row.to_dict())

# df with split datasets
split_df = pd.DataFrame(rnnlib.final_list(by_category, args))
split_df["split"].value_counts()

split_df.title = split_df.title.apply(rnnlib.preprocess_text)
split_df.head()

# Vocabulary instance
category_vocab = Vocabulary()
for index, row in df.iterrows():
    category_vocab.add_token(row.category)

# Get word counts
word_counts = Counter()
for title in split_df.title:
    for token in title.split(" "):
        if token not in string.punctuation:
            word_counts[token] += 1

# Create SequenceVocabulary instance
title_vocab = SequenceVocabulary()
for word, word_count in word_counts.items():
    if word_count >= args.cutoff:
        title_vocab.add_token(word)

index = title_vocab.lookup_token("general")

# Vectorizer instance
vectorizer = NewsVectorizer.from_dataframe(split_df, cutoff=args.cutoff)

# Dataset instance
dataset = NewsDataset.load_dataset_and_make_vectorizer(df=split_df, cutoff=args.cutoff)
# Load vectorizer
with open(args.vectorizer_file) as fp:
    vectorizer = NewsVectorizer.from_serializable(json.load(fp))

# Load the model
model = NewsModel(embedding_dim=args.embedding_dim, 
                  num_embeddings=len(vectorizer.title_vocab), 
                  rnn_hidden_dim=args.rnn_hidden_dim,
                  hidden_dim=args.hidden_dim,
                  output_dim=len(vectorizer.category_vocab),
                  num_layers=args.num_layers,
                  bidirectional=args.bidirectional,
                  dropout_p=args.dropout_p, 
                  pretrained_embeddings=None, 
                  padding_idx=vectorizer.title_vocab.mask_index)
model.load_state_dict(torch.load(args.model_state_file))

# Initialize
inference = Inference(model=model, vectorizer=vectorizer, device=args.device)

def classify_title(title):
    #title = input("Enter a title to classify: ")
    infer_df = pd.DataFrame([title], columns=['title'])
    infer_df.title = infer_df.title.apply(rnnlib.preprocess_text)
    infer_dataset = InferenceDataset(infer_df, vectorizer)
    results = inference.predict_category(dataset=infer_dataset)
    return results
