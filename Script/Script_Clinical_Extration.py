"""
Script 02: Clinical Entity Extraction (spaCy)
Context: Data Extraction from 65 Selected Studies
Description: Extracts frequent clinical nouns and adjectives from physical examination domains.
"""

import pandas as pd
import spacy
from collections import Counter

# Load English clinical language model
nlp = spacy.load("en_core_web_sm")

# Load the extraction database
df = pd.read_csv ("clinical_extractions.xlsx")

def extract_clinical_terms(text_series):
    """Tokenises text and filters for nouns and adjectives."""
    combined_text = " ".join(text_series.fillna("").astype(str))
    doc = nlp(combined_text.lower())
    
    # Filtering for relevant POS tags and removing stop words
    keywords = [
        token.lemma_ for token in doc 
        if token.pos_ in ["NOUN", "ADJ"] 
        and not token.is_stop 
        and len(token.text) > 2
    ]
    return Counter(keywords).most_common(30)

# Process domains: Inspection, Palpation, Auscultation, Conclusions
domains = ['Inspection01', 'Palpation01', 'Auscultation01', 'Conclusions01']
results = {domain: extract_clinical_terms(df[domain]) for domain in domains if domain in df.columns}

# Export findings
results_df = pd.DataFrame(results)
results_df.to_csv("clinical_domain_extraction_results.csv", index=False)
print("Clinical extraction completed.")
