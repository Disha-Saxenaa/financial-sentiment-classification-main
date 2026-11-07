import pandas as pd
import os

# Define the mapping from the numeric score to sentiment labels
def map_score_to_sentiment(score):
    """
    Map a numeric sentiment score to our three-class system.
    Scores range from -1 (very negative) to 1 (very positive).
    """
    if pd.isna(score):
        return "neutral"
    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

def process_fiqa_file(file_path):
    """
    Read a FiQA parquet file, filter for headlines, and convert req format.
    """
    print(f"  Processing: {os.path.basename(file_path)}")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    print(f"    Original rows: {len(df)}")
    
    # Filter for headlines only
    df_headlines = df[df['type'] == 'headline'].copy()
    print(f"    Headlines only: {len(df_headlines)}")
    
    if len(df_headlines) == 0:
        return pd.DataFrame(columns=['text', 'sentiment'])
    
    # Rename 'sentence' to 'text'
    df_headlines = df_headlines.rename(columns={'sentence': 'text'})
    
    # Apply sentiment mapping
    df_headlines['sentiment'] = df_headlines['score'].apply(map_score_to_sentiment)
    
    # Keep only the columns we need
    df_final = df_headlines[['text', 'sentiment']]
    
    return df_final

# --- Main Execution ---
print("="*60)
print("Processing FiQA Sentiment Dataset")
print("="*60)

files = [
    "train.parquet",
    "valid.parquet",
    "test.parquet"
]

all_dfs = []
for file in files:
    if os.path.exists(file):
        df_processed = process_fiqa_file(file)
        all_dfs.append(df_processed)
    else:
        print(f"  ⚠️ Warning: File not found - {file}")

# Combine all DataFrames
if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    print("\n" + "="*60)
    print("Final Dataset Summary")
    print("="*60)
    print(f"Total samples: {len(final_df)}")
    print("\nClass distribution:")
    print(final_df['sentiment'].value_counts())
    print("\nPercentages:")
    print(final_df['sentiment'].value_counts(normalize=True).mul(100).round(1))
    
    # Save to CSV
    final_df.to_csv('fiqa_sentiment.csv', index=False)
    print("\n Dataset saved to 'fiqa_sentiment.csv'")
else:
    print("\n No data processed. Please ensure the .parquet files are in the same directory.")