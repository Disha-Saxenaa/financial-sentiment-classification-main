import pandas as pd
import requests
from io import StringIO
# NOT FOR SUBMISSION
# Financial PhraseBank is also available as a CSV on GitHub
# This is a cleaned version of the dataset
url = "https://raw.githubusercontent.com/TurkuNLP/financial-phrasebank/refs/heads/main/FinancialPhraseBank-v1.0/Sentences_50Agree.txt"

response = requests.get(url)
content = response.text

# Parse the data
data = []
for line in content.strip().split('\n'):
    if not line.strip():
        continue
    # Format: "sentence @label@"
    parts = line.rsplit('@', 1)
    if len(parts) == 2:
        text = parts[0].strip()
        label = parts[1].strip()
        # Convert label: 0=negative, 1=neutral, 2=positive
        label_map = {"0": "negative", "1": "neutral", "2": "positive"}
        data.append({
            'text': text,
            'label': int(label),
            'sentiment': label_map.get(label, "unknown")
        })

df = pd.DataFrame(data)

print(f"✅ Loaded {len(df)} samples")
print("\nClass distribution:")
print(df['sentiment'].value_counts())

# Save to CSV
df.to_csv("financial_phrasebank.csv", index=False)
print(f"\n✅ Dataset saved to financial_phrasebank.csv")

# Show samples
print("\nSample rows:")
print(df.head())