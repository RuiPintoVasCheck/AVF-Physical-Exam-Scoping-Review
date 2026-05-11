"""
Script 01: BeRTopic Analysis for Search String Expansion
Context: Scoping Review on AVF Physical Examination
Description: Applies BeRT embeddings and c-TF-IDF to identify clinical descriptors.
"""

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load PubMed abstracts
with open('2025_10_29_Pubmed-AVF-set.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

abstracts = [line.replace('AB  - ', '').strip() for line in lines if line.startswith('AB  -')]

# Initialise BeRT model (Academic Standard)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Train BERTopic model
topic_model = BERTopic(
    embedding_model=sentence_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probs = topic_model.fit_transform(abstracts)

# Extract and save topic-representative words (c-TF-IDF)
topic_info = topic_model.get_topic_info()
topic_info.to_csv("appendix_nlp_topic_words.csv", index=False)

print("BERTopic analysis completed. Results exported to 'appendix_nlp_topic_words.csv'.")
