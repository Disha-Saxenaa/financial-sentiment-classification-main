import pandas as pd
# NOT FOR SUBMISSION. THIS CODE IS REDUNDANT.
def parse_financial_phrasebank(filename):
    """Parse the Financial PhraseBank text file directly with string labels"""
    data = []
    
    with open(filename, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split on the last @ symbol
            if '@' in line:
                parts = line.rsplit('@', 1)
                if len(parts) == 2:
                    text = parts[0].strip()
                    label = parts[1].strip()
                    
                    # Only keep valid labels
                    if label in ['negative', 'neutral', 'positive']:
                        data.append({
                            'text': text,
                            'sentiment': label
                        })
    
    return pd.DataFrame(data)

# Parse the dataset
print("Parsing Financial PhraseBank...")
df = parse_financial_phrasebank("Sentences_50Agree.txt")

print(f"\n✅ Loaded {len(df)} samples")
print(f"✅ Columns: {df.columns.tolist()}")

print("\n" + "="*60)
print("CLASS DISTRIBUTION:")
print("="*60)
print(df['sentiment'].value_counts())
print(f"\nPercentages:")
print(df['sentiment'].value_counts(normalize=True).mul(100).round(1))

print("\n" + "="*60)
print("SAMPLE ROWS (First 10):")
print("="*60)
print(df.head(10))

print("\n" + "="*60)
print("EXAMPLES OF EACH CLASS:")
print("="*60)

for sentiment in ['positive', 'neutral', 'negative']:
    examples = df[df['sentiment'] == sentiment]
    if len(examples) > 0:
        print(f"\n{sentiment.upper()} (sample):")
        print(f"  Text: {examples.iloc[0]['text'][:150]}...")
        print(f"  Label: {examples.iloc[0]['sentiment']}")

# Save to CSV
df.to_csv("financial_phrasebank.csv", index=False)
print(f"\n Dataset saved to financial_phrasebank.csv")

# Create a smaller subset for quick testing
df_small = df.head(50)
df_small.to_csv("financial_phrasebank_small.csv", index=False)
print(f" Small subset (50 samples) saved to financial_phrasebank_small.csv")