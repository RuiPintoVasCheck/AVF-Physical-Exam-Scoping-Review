import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import os
import re

# =========================
# SETTINGS
# =========================
file_path = r"C:\Users\User\Downloads\clinical_extractions.xlsx"
target_cols = ['Inspection01', 'Palpation01', 'Auscultation01', 'Conclusions01']
top_n = 20
ngram_range = (1, 3)   # (1,1)=single words only, (1,3)=words + phrases
output_excel = r"C:\Users\User\Downloads\nlp_results.xlsx"
chart_folder = r"C:\Users\User\Downloads\nlp_charts"

# =========================
# LOAD NLP MODEL
# =========================
nlp = spacy.load("en_core_web_sm")

# =========================
# FUNCTIONS
# =========================
def safe_filename(name):
    return re.sub(r'[^A-Za-z0-9_-]+', '_', str(name))

def clinical_clean(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    if not text:
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
        and len(token.text) > 2
    ]
    return " ".join(tokens)

def get_top_terms(text_series, top_n=20, ngram_range=(1, 3)):
    cleaned = text_series.fillna("").astype(str).apply(clinical_clean)
    cleaned = cleaned[cleaned.str.strip() != ""]

    if cleaned.empty:
        return pd.DataFrame(columns=["term", "frequency"])

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words="english"
    )
    X = vectorizer.fit_transform(cleaned)

    terms = vectorizer.get_feature_names_out()
    freqs = X.sum(axis=0).A1

    result = pd.DataFrame({
        "term": terms,
        "frequency": freqs
    }).sort_values("frequency", ascending=False).head(top_n)

    return result.reset_index(drop=True)

def plot_top_terms(df_terms, title, save_path=None):
    if df_terms.empty:
        print(f"No data to plot for: {title}")
        return

    plt.figure(figsize=(12, 8))
    plt.barh(df_terms["term"][::-1], df_terms["frequency"][::-1])
    plt.title(title, fontsize=14)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Clinical Terms / Phrases", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Chart saved: {save_path}")

    plt.show()

def analyze_combined_consensus(df, target_cols, top_n=20, ngram_range=(1, 3)):
    df = df.copy()
    df["combined_text"] = df[target_cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["clean_findings"] = df["combined_text"].apply(clinical_clean)
    return get_top_terms(df["clean_findings"], top_n=top_n, ngram_range=ngram_range)

def analyze_each_column(df, target_cols, top_n=20, ngram_range=(1, 3)):
    results = {}
    for col in target_cols:
        results[col] = get_top_terms(df[col], top_n=top_n, ngram_range=ngram_range)
    return results

# =========================
# MAIN
# =========================
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    print("Try putting the file in the same folder as this script and use file_path = '1.xlsx'")
else:
    df = pd.read_excel(file_path)

    missing = [col for col in target_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns are: {df.columns.tolist()}")
    else:
        os.makedirs(chart_folder, exist_ok=True)

        print(f"Cleaning and analyzing columns: {target_cols}")

        # 1. Combined consensus analysis
        consensus_df = analyze_combined_consensus(
            df,
            target_cols=target_cols,
            top_n=top_n,
            ngram_range=ngram_range
        )

        print("\nTOP CONSENSUS MANEUVERS/FINDINGS (COMBINED):")
        print(consensus_df)

        plot_top_terms(
            consensus_df,
            title="Consensus of Physical Exam Findings (Combined NLP Analysis)",
            save_path=os.path.join(chart_folder, "combined_consensus.png")
        )

        # 2. Per-column analysis
        per_column_results = analyze_each_column(
            df,
            target_cols=target_cols,
            top_n=top_n,
            ngram_range=ngram_range
        )

        for col, result_df in per_column_results.items():
            print(f"\nTOP TERMS IN COLUMN: {col}")
            print(result_df)

            plot_top_terms(
                result_df,
                title=f"Top Terms in Column: {col}",
                save_path=os.path.join(chart_folder, f"{safe_filename(col)}_top_terms.png")
            )

        # 3. Save everything to Excel
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            consensus_df.to_excel(writer, sheet_name="combined_consensus", index=False)

            summary_rows = []
            for col, result_df in per_column_results.items():
                sheet_name = safe_filename(col)[:31]
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)

                temp = result_df.copy()
                temp.insert(0, "column", col)
                summary_rows.append(temp)

            if summary_rows:
                summary_df = pd.concat(summary_rows, ignore_index=True)
                summary_df.to_excel(writer, sheet_name="all_columns_summary", index=False)

        print(f"\nDone. Excel results saved to: {output_excel}")
        print(f"Charts saved in folder: {chart_folder}")
