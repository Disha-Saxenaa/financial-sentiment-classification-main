import ollama
import pandas as pd
import time
from datetime import datetime
import sys
import argparse
import os
import re

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONFIGURATION - EDIT THESE FOR EACH RUN
# ============================================================
CONFIG = {
    "model_name": "hf.co/Mungert/Fin-R1-GGUF:latest",
    "dataset_file": "financial_phrasebank_small.csv",
    "output_prefix": "experiment_results",
    "max_samples": None,  # Set to e.g., 100 to limit samples
    "techniques": ["zero_shot", "few_shot", "self_consistency", "hybrid"],  # Default
}

# ============================================================
# PROMPT TEMPLATES
# ============================================================
PROMPT_TEMPLATES = {
    "zero_shot": """
You are a financial sentiment analyst. Classify the sentiment of the following financial news text as 'positive', 'negative', or 'neutral'.

Text: {text}

Sentiment:""",

    "few_shot": """
You are a financial sentiment analyst. Classify the sentiment of the following financial news text as 'positive', 'negative', or 'neutral'.

Examples:
Text: "The company reported record profits this quarter."
Sentiment: positive

Text: "The stock market crashed due to inflation fears."
Sentiment: negative

Text: "The board announced a routine quarterly review."
Sentiment: neutral

Now classify this text:
Text: {text}
Sentiment:""",

    "self_consistency": """
You are a financial sentiment analyst. Think step-by-step about the sentiment of the following financial news text.Think step-by-step about the sentiment of the following financial news text .Classify the sentiment of the following financial news text as 'positive', 'negative', or 'neutral'.

Text: {text}

Sentiment:""",

    "hybrid": """
You are a financial sentiment analyst. You have been given the following instructions from the dataset annotators:

Instructions for annotators:
- Sentiment is the opinion or feeling expressed in the text towards the financial entity.
- Positive: The text expresses a favorable or optimistic view.
- Negative: The text expresses an unfavorable or pessimistic view.
- Neutral: The text expresses no clear opinion or contains only factual information.

Examples:
Text: "The company reported record profits this quarter."
Sentiment: positive

Text: "The stock market crashed due to inflation fears."
Sentiment: negative

Text: "The board announced a routine quarterly review."
Sentiment: neutral

Now, carefully analyze this text and classify its sentiment as 'positive', 'negative', or 'neutral' using the instructions above.

Text: {text}

Sentiment:"""
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def call_llm(prompt, model, temperature=0, max_retries=2):
    """Send prompt to Ollama with specified temperature"""
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 1000
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"    ERROR (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                print(f"    Retrying in 5 seconds...")
                time.sleep(5)
            else:
                return None

def parse_response(response):
    """Extract sentiment from LLM response - robust version with markdown support"""
    if response is None or pd.isna(response):
        return "unknown"
    
    response_str = str(response)
    response_lower = response_str.lower()
    
    # Handle the case where the response is just the sentiment word
    if response_lower.strip() in ["positive", "negative", "neutral"]:
        return response_lower.strip()
    
    # Look for patterns like "Sentiment: positive" or "Sentiment = positive" (with markdown)
    patterns = [
        r'sentiment\s*[:=]\s*\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'final sentiment\s*[:=]\s*\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'overall sentiment\s*[:=]\s*\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'sentiment\s+is\s+\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'classified\s+as\s+\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'classification\s*[:=]\s*\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'final answer\s*[:=]\s*\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'answer\s*[:=]\s*\*{0,2}(positive|negative|neutral)\*{0,2}',
        r'\\boxed\s*\{\s*\*{0,2}(positive|negative|neutral)\*{0,2}\s*\}',
        r'boxed\s*\{\s*\*{0,2}(positive|negative|neutral)\*{0,2}\s*\}',
        r'(?:^|\n)\s*(?:final|overall)?\s*sentiment\s*(?:is|:)?\s*\*{0,2}(positive|negative|neutral)\*{0,2}\s*[.!]?$',

        # NEW PATTERNS FOR QUOTED SENTIMENTS:
        # "I would classify the sentiment of this financial news text as ""Neutral""."
        r'(?:classify|classifying|classify the sentiment of this financial news text as)\s+["\'](positive|negative|neutral)["\']',
        r'(?:classify|classification)\s+["\'](positive|negative|neutral)["\']',
        r'sentiment\s+as\s+["\'](positive|negative|neutral)["\']',
        r'sentiment\s+is\s+["\'](positive|negative|neutral)["\']',
        r'would classify this as\s+["\'](positive|negative|neutral)["\']',
        r'would classify the sentiment as\s+["\'](positive|negative|neutral)["\']',
        r'would be classified as\s+["\'](positive|negative|neutral)["\']',
        r'^["\'](positive|negative|neutral)["\']$',
        
        # Catch patterns like: "The sentiment is: positive"
        r'sentiment\s*[:=]\s*(positive|negative|neutral)',
        r'sentiment\s+is\s+(positive|negative|neutral)',

        # NEW PATTERNS FOR fp100:
        r'(?:classify|classifying)\s+the\s+sentiment\s+of\s+this\s+financial\s+news\s+text\s+as\s+["\'](positive|negative|neutral)["\']',
        r'(?:classify|classifying)\s+the\s+sentiment\s+as\s+["\'](positive|negative|neutral)["\']',
        r'would\s+classify\s+this\s+as\s+["\'](positive|negative|neutral)["\']',
        r'would\s+be\s+classified\s+as\s+["\'](positive|negative|neutral)["\']',
        r'would\s+classify\s+the\s+sentiment\s+as\s+["\']?(positive|negative|neutral)["\']?',
        r'^["\'](positive|negative|neutral)["\']',
        r'classify\s+this\s+news\s+as\s+["\'](positive|negative|neutral)["\']',
    ]
    match_found = False
    
    for pattern in patterns:
        match = re.search(pattern, response_lower, re.IGNORECASE)
        if match:
            match_found = True
            return match.group(1).lower()

    # ============================================================
    # NEW BLOCK: Only runs if patterns above found nothing
    # ============================================================
    if not match_found:
        classification_phrases = [
            "would classify",
            "would be classified",
            "classify the sentiment",
            "classification"
        ]
        
        if any(phrase in response_lower for phrase in classification_phrases):
            # Look for "as 'positive'"
            quote_match = re.search(r"as\s+['\"]?(positive|negative|neutral)['\"]?", response_lower)
            if quote_match:
                return quote_match.group(1).lower()
            
            # Fallback: look for the sentiment word in the classification sentence
            sentences = re.split(r'[.!?]+', response_lower)
            for sentence in sentences:
                if any(phrase in sentence for phrase in classification_phrases):
                    for word in ["positive", "negative", "neutral"]:
                        if word in sentence:
                            return word
    
    # Find the last occurrence of any sentiment word near the end
    sentences = re.split(r'[.!?]+', response_lower)
    for sentence in reversed(sentences[-3:]):
        sentiment_match = re.search(r'(?:sentiment|classification|answer)\s+is\s+\*{0,2}(positive|negative|neutral)\*{0,2}', sentence)
        if sentiment_match:
            return sentiment_match.group(1)
        for word in ["positive", "negative", "neutral"]:
            if word in sentence and ("sentiment" in sentence or "classification" in sentence):
                return word
    
    # Count occurrences
    pos_count = response_lower.count("positive")
    neg_count = response_lower.count("negative")
    neu_count = response_lower.count("neutral")
    
    if pos_count >= 2 and pos_count >= neg_count and pos_count >= neu_count:
        return "positive"
    elif neg_count >= 2 and neg_count >= pos_count and neg_count >= neu_count:
        return "negative"
    elif neu_count >= 2 and neu_count >= pos_count and neu_count >= neg_count:
        return "neutral"
    
    return "unknown"

def run_experiment_single(df, technique, prompt_template, model_name, temperature=0):
    """Run one experiment with a specific technique (single pass)"""
    results = []
    print(f"\n=== Running {technique} ===")
    
    for idx, row in df.iterrows():
        text = row['text']
        true_label = row['sentiment']
        
        print(f"  Sample {idx+1}/{len(df)}: ", end="", flush=True)
        
        prompt = prompt_template.format(text=text)
        response = call_llm(prompt, model_name, temperature=temperature)
        predicted = parse_response(response)
        
        results.append({
            'statement': text,
            'true_sentiment': true_label,
            'predicted_sentiment': predicted,
            'raw_response': response,
            'technique': technique
        })
        
        status = "✅" if predicted == true_label else "❌"
        print(f"{status} True={true_label}, Predicted={predicted}", flush=True)
        time.sleep(0.5)
    
    return pd.DataFrame(results)

def run_experiment_self_consistency(df, technique, prompt_template, model_name, num_runs=3):
    """Run self-consistency: multiple passes with temperature > 0, then majority vote"""
    print(f"\n=== Running {technique} (with Self-Consistency, {num_runs} runs) ===")
    
    all_results = []
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}:")
        temperature = 0.1 + (run * 0.1)
        results = []
        
        for idx, row in df.iterrows():
            text = row['text']
            true_label = row['sentiment']
            
            # Show progress immediately
            print(f"    Sample {idx+1}/{len(df)}: ", end="", flush=True)
            
            prompt = prompt_template.format(text=text)
            response = call_llm(prompt, model_name, temperature=temperature)
            predicted = parse_response(response)
            
            # Show result immediately
            status = "✅" if predicted == true_label else "❌"
            print(f"{status} True={true_label}, Predicted={predicted}", flush=True)
            
            results.append({
                'id': idx,
                'true_sentiment': true_label,
                'predicted_sentiment': predicted,
                'run': run,
                'raw_response': response  # ADD THIS - store the raw response
            })
            time.sleep(0.5)
        
        all_results.append(pd.DataFrame(results))
        print(f"    Completed run {run+1}")
    
    combined = pd.concat(all_results, ignore_index=True)
    final_results = []
    
    for idx in df.index:
        sample_predictions = combined[combined['id'] == idx]
        true_label = sample_predictions['true_sentiment'].iloc[0]
        
        vote_counts = sample_predictions['predicted_sentiment'].value_counts()
        if len(vote_counts) > 0:
            predicted = vote_counts.index[0]
        else:
            predicted = "unknown"
        
        # Get the first raw response from this sample (from run 0)
        first_response = combined[(combined['id'] == idx) & (combined['run'] == 0)]['raw_response'].iloc[0] if len(combined[(combined['id'] == idx) & (combined['run'] == 0)]) > 0 else f"Self-consistency vote from {num_runs} runs"
        
        final_results.append({
            'statement': df.iloc[idx]['text'],
            'true_sentiment': true_label,
            'predicted_sentiment': predicted,
            'raw_response': first_response,
            'technique': technique
        })
    
    return pd.DataFrame(final_results)

# ============================================================
# MAIN FUNCTION
# ============================================================

def run_experiment(model_name=None, dataset_file=None, output_prefix=None, 
                   max_samples=None, techniques=None):
    """
    Run the full experiment with the given configuration.
    """
    
    # Use config defaults if not provided
    if model_name is None:
        model_name = CONFIG["model_name"]
    if dataset_file is None:
        dataset_file = CONFIG["dataset_file"]
    if output_prefix is None:
        output_prefix = CONFIG["output_prefix"]
    if max_samples is None:
        max_samples = CONFIG["max_samples"]
    if techniques is None:
        techniques = CONFIG["techniques"]
    
    # Validate techniques
    valid_techniques = list(PROMPT_TEMPLATES.keys())
    for tech in techniques:
        if tech not in valid_techniques:
            print(f"❌ Invalid technique: {tech}")
            print(f"   Available: {valid_techniques}")
            return None
    
    # ============================================================
    # LOAD DATASET
    # ============================================================
    
    print("="*60)
    print(f"EXPERIMENT STARTING")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_file}")
    print(f"Techniques: {techniques}")
    
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return None
    
    df = pd.read_csv(dataset_file)
    
    # Handle column names
    if 'text' not in df.columns and 'sentence' in df.columns:
        df = df.rename(columns={'sentence': 'text'})
    
    if 'sentiment' not in df.columns and 'label' in df.columns:
        if df['label'].dtype in ['int64', 'float64']:
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            df['sentiment'] = df['label'].map(label_map)
        else:
            df = df.rename(columns={'label': 'sentiment'})
    
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        print("❌ Dataset must have 'text' and 'sentiment' columns")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    if max_samples and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
        print(f"Using random subset of {max_samples} samples")
    
    print(f"Loaded {len(df)} samples")
    print(f"Classes: {df['sentiment'].unique()}")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}")
    
    # ============================================================
    # TEST OLLAMA CONNECTION
    # ============================================================
    
    print("\nTesting Ollama connection...")
    try:
        test_response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            options={"temperature": 0}
        )
        print(f"✅ Ollama is running with {model_name}!")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("Please make sure Ollama is running with: ollama serve")
        return None
    
    # ============================================================
    # RUN EXPERIMENTS (Only for specified techniques)
    # ============================================================
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    
    all_results = []
    
    for technique in techniques:
        if technique == "self_consistency":
            results = run_experiment_self_consistency(
                df, technique, PROMPT_TEMPLATES[technique], model_name
            )
        else:
            results = run_experiment_single(
                df, technique, PROMPT_TEMPLATES[technique], model_name
            )
        
        all_results.append(results)
        
        tech_file = f"{output_prefix}_{timestamp}_{dataset_name}_{technique}.csv"
        results.to_csv(tech_file, index=False)
        print(f"  💾 Saved to {tech_file}")
    
    # ============================================================
    # COMBINE AND SAVE FINAL RESULTS
    # ============================================================
    
    final_df = pd.concat(all_results, ignore_index=True)
    column_order = ['statement', 'true_sentiment', 'predicted_sentiment', 'technique', 'raw_response']
    final_df = final_df[column_order]
    
    final_file = f"{output_prefix}_{timestamp}_{dataset_name}_final.csv"
    final_df.to_csv(final_file, index=False)
    
    print(f"\n✅ All experiments complete!")
    print(f"✅ Final results saved to {final_file}")
    print(f"Total runs: {len(final_df)}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    
    print("\n" + "="*60)
    print("QUICK SUMMARY - Accuracy per technique:")
    print("="*60)
    
    summary_data = []
    for technique in techniques:
        subset = final_df[final_df['technique'] == technique]
        correct = (subset['true_sentiment'] == subset['predicted_sentiment']).sum()
        total = len(subset)
        accuracy = correct / total * 100
        print(f"{technique}: {accuracy:.1f}% ({correct}/{total})")
        summary_data.append({
            'technique': technique,
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = f"{output_prefix}_{timestamp}_{dataset_name}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n💾 Summary saved to {summary_file}")
    
    return final_df

# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM prompt engineering experiments")
    parser.add_argument("--model", "-m", help="Ollama model name")
    parser.add_argument("--dataset", "-d", help="Dataset CSV file path")
    parser.add_argument("--output", "-o", help="Output file prefix")
    parser.add_argument("--max", "-n", type=int, help="Maximum number of samples to use")
    parser.add_argument("--techniques", "-t", nargs="+", 
                        choices=["zero_shot", "few_shot", "self_consistency", "hybrid"],
                        help="Which techniques to run (space separated)")
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    model_name = args.model if args.model else CONFIG["model_name"]
    dataset_file = args.dataset if args.dataset else CONFIG["dataset_file"]
    output_prefix = args.output if args.output else CONFIG["output_prefix"]
    max_samples = args.max if args.max is not None else CONFIG["max_samples"]
    techniques = args.techniques if args.techniques else CONFIG["techniques"]
    
    run_experiment(
        model_name=model_name,
        dataset_file=dataset_file,
        output_prefix=output_prefix,
        max_samples=max_samples,
        techniques=techniques
    )