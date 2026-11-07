import pandas as pd
import os

def parse_financial_phrasebank(filename):
    """Parse the Financial PhraseBank text file into a DataFrame."""
    data = []
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filename, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if '@' in line:
                        parts = line.rsplit('@', 1)
                        if len(parts) == 2:
                            text = parts[0].strip()
                            label = parts[1].strip()
                            if label in ['negative', 'neutral', 'positive']:
                                data.append({
                                    'text': text,
                                    'sentiment': label
                                })
            print(f"  Parsed {filename} with {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    return pd.DataFrame(data)

def deduplicate_datasets():
    """Create unique versions of each agreement level."""
    
    print("="*60)
    print("Deduplicating Financial PhraseBank")
    print("="*60)
    
    # Files in increasing agreement order
    files = [
        "Sentences_50Agree.txt",
        "Sentences_66Agree.txt", 
        "Sentences_75Agree.txt",
        "Sentences_AllAgree.txt"
    ]
    
    # Parse all files
    dfs = {}
    for file in files:
        print(f"\nParsing {file}...")
        dfs[file] = parse_financial_phrasebank(file)
        print(f"  Loaded {len(dfs[file])} samples")
    
    # Create unique versions
    print("\n" + "="*60)
    print("Creating Unique Datasets")
    print("="*60)
    
    # 1. AllAgree: keep everything
    df_all = dfs["Sentences_AllAgree.txt"].copy()
    df_all.to_csv("FinancialPhraseBank_AllAgree.csv", index=False)
    print(f"\n✅ FinancialPhraseBank_AllAgree.csv: {len(df_all)} samples (100% agreement)")
    
    # 2. 75Agree: remove sentences that are in AllAgree
    df_75 = dfs["Sentences_75Agree.txt"].copy()
    # Create a set of texts from AllAgree for fast lookup
    all_texts = set(df_all['text'])
    df_75_unique = df_75[~df_75['text'].isin(all_texts)]
    df_75_unique.to_csv("FinancialPhraseBank_75Agree.csv", index=False)
    print(f"✅ FinancialPhraseBank_75Agree.csv: {len(df_75_unique)} samples (75% agreement, not in 100%)")
    
    # 3. 66Agree: remove sentences that are in 75Agree
    df_66 = dfs["Sentences_66Agree.txt"].copy()
    # Create a set of texts from 75Agree (including its unique ones)
    # We need the original 75Agree dataset, not just the unique one
    df_75_original = dfs["Sentences_75Agree.txt"]
    seventyfive_texts = set(df_75_original['text'])
    df_66_unique = df_66[~df_66['text'].isin(seventyfive_texts)]
    df_66_unique.to_csv("FinancialPhraseBank_66Agree.csv", index=False)
    print(f"✅ FinancialPhraseBank_66Agree.csv: {len(df_66_unique)} samples (66% agreement, not in 75%)")
    
    # 4. 50Agree: remove sentences that are in 66Agree
    df_50 = dfs["Sentences_50Agree.txt"].copy()
    # Use the original 66Agree dataset
    df_66_original = dfs["Sentences_66Agree.txt"]
    sixtysix_texts = set(df_66_original['text'])
    df_50_unique = df_50[~df_50['text'].isin(sixtysix_texts)]
    df_50_unique.to_csv("FinancialPhraseBank_50Agree.csv", index=False)
    print(f"✅ FinancialPhraseBank_50Agree.csv: {len(df_50_unique)} samples (50% agreement, not in 66%)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    summary = [
        ("50Agree", df_50_unique),
        ("66Agree", df_66_unique),
        ("75Agree", df_75_unique),
        ("AllAgree", df_all)
    ]
    
    for name, df in summary:
        print(f"\n{name}:")
        print(f"  Total samples: {len(df)}")
        print(f"  Class distribution:")
        print(df['sentiment'].value_counts())
        print(f"  Class percentages:")
        print(df['sentiment'].value_counts(normalize=True).mul(100).round(1))
    
    print("\n" + "="*60)
    print("✅ All files saved successfully!")

if __name__ == "__main__":
    deduplicate_datasets()