import pandas as pd
import ast

def parse_sentiment(value):
    """
    Extract overall sentiment from Decisions column.
    Returns None if there are multiple entities (regardless of sentiment agreement).
    """
    try:
        d = ast.literal_eval(value)
        sentiments = list(d.values())
        
        # If there's only ONE entity, return its sentiment
        if len(sentiments) == 1:
            return sentiments[0]
        
        # If there are MULTIPLE entities, return None
        return None
    except:
        return None

print("Loading SEntFiN.csv")
df = pd.read_csv('SEntFiN.csv')

print("Parsing sentiments and filtering rows...")
df['sentiment'] = df['Decisions'].apply(parse_sentiment)

# Remove rows where sentiment is None
df = df.dropna(subset=['sentiment'])

df = df.rename(columns={'Title': 'text'})
df = df[['text', 'sentiment']]

print(f"\n Processed {len(df)} samples")
print("\nClass distribution:")
print(df['sentiment'].value_counts())
print(f"\nPercentages:")
print(df['sentiment'].value_counts(normalize=True).mul(100).round(1))

# Show sample rows
print("\nSample rows:")
print(df.head(10))

# Save full dataset
df.to_csv('SEntFiN_full.csv', index=False)
print("\n Full dataset saved to SEntFiN_full.csv")