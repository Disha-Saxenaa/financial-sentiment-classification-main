import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score
)
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import argparse
from datetime import datetime

# ============================================================
# CONFIGURATION
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
You are a financial sentiment analyst. Classify the sentiment of the following financial news text as 'positive', 'negative', or 'neutral'.

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

DATASET_MAP = {
    'fp50': 'FinancialPhraseBank_50Agree',
    'fp66': 'FinancialPhraseBank_66Agree',
    'fp75': 'FinancialPhraseBank_75Agree',
    'fp100': 'FinancialPhraseBank_100Agree',
    'fiqa': 'FiQA'
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_filename(filename):
    """Parse filename to extract model, dataset, technique."""
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    techniques = ['zero_shot', 'few_shot', 'self_consistency', 'hybrid']
    tech = None
    for t in techniques:
        if base.endswith('_' + t):
            tech = t
            base = base[:-len(t)-1]
            break
    if tech is None:
        raise ValueError(f"Could not parse technique from {filename}")
    
    datasets = ['fp50', 'fp66', 'fp75', 'fp100', 'fiqa']
    ds = None
    for d in datasets:
        if base.endswith('_' + d):
            ds = d
            base = base[:-len(d)-1]
            break
    if ds is None:
        raise ValueError(f"Could not parse dataset from {filename}")
    
    model = base
    return model, ds, tech

def load_all_results(directory):
    """Load all CSV files from directory and combine into a single DataFrame."""
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data = []
    for f in all_files:
        try:
            model, dataset, technique = parse_filename(f)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
        df = pd.read_csv(os.path.join(directory, f))
        df['model'] = model
        df['dataset'] = dataset
        df['technique'] = technique
        data.append(df)
    if not data:
        raise ValueError("No valid result files found.")
    return pd.concat(data, ignore_index=True)

def compute_metrics(df):
    """Compute precision, recall, f1 per class, macro, weighted, accuracy, confusion matrix."""
    labels = ['positive', 'negative', 'neutral']
    y_true = df['true_sentiment']
    y_pred = df['predicted_sentiment']
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )[2]
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )[2]
    accuracy = accuracy_score(y_true, y_pred)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        'accuracy': accuracy,
        'precision': dict(zip(labels, precision)),
        'recall': dict(zip(labels, recall)),
        'f1': dict(zip(labels, f1)),
        'support': dict(zip(labels, support)),
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'confusion_matrix': cm,
        'labels': labels
    }
    return metrics

def format_compliance(df):
    """This section is redundant and to be removed."""
    def is_compliant(text):
        if pd.isna(text):
            return False
        t = str(text).strip().lower()
        if t in ['positive', 'negative', 'neutral']:
            return True
        if re.match(r'^(positive|negative|neutral)[.!]?$', t):
            return True
        words = t.split()
        if len(words) <= 10:
            if 'positive' in t or 'negative' in t or 'neutral' in t:
                return True
        return False
    
    df['compliant'] = df['raw_response'].apply(is_compliant)
    return df['compliant'].mean() * 100

def token_efficiency(technique):
    """Compute average prompt length for a given technique."""
    template = PROMPT_TEMPLATES.get(technique)
    if template is None:
        return np.nan
    sample_text = "The company reported record profits."
    prompt = template.format(text=sample_text)
    return len(prompt) / 4

def count_tokens(text):
    """Approximate token count using 4 characters per token."""
    if text is None or pd.isna(text):
        return 0
    return len(str(text)) / 4

def aggregate_by(df, group_cols):
    """Aggregate metrics by given columns."""
    results = []
    for name, group in df.groupby(group_cols):
        if not isinstance(name, tuple):
            name = (name,)
        metrics = compute_metrics(group)
        compliant = format_compliance(group)
        
        # Get technique for token efficiency if available
        tech = name[-1] if 'technique' in group_cols else None
        if tech is not None:
            # Input tokens (from prompt template)
            input_tokens = token_efficiency(tech)
            
            # Output tokens (from raw responses)
            avg_output_tokens = group['raw_response'].apply(count_tokens).mean()
            
            # Token efficiency ratio: output / (input + output) * 100
            if input_tokens + avg_output_tokens > 0:
                token_eff_ratio = avg_output_tokens / (input_tokens + avg_output_tokens) * 100
            else:
                token_eff_ratio = 0
        else:
            input_tokens = np.nan
            avg_output_tokens = np.nan
            token_eff_ratio = np.nan
        
        row = {
            'group': ' | '.join(str(n) for n in name),
            **{k: name[i] for i, k in enumerate(group_cols)},
            'accuracy': metrics['accuracy'],
            'precision_positive': metrics['precision']['positive'],
            'precision_negative': metrics['precision']['negative'],
            'precision_neutral': metrics['precision']['neutral'],
            'recall_positive': metrics['recall']['positive'],
            'recall_negative': metrics['recall']['negative'],
            'recall_neutral': metrics['recall']['neutral'],
            'f1_positive': metrics['f1']['positive'],
            'f1_negative': metrics['f1']['negative'],
            'f1_neutral': metrics['f1']['neutral'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'format_compliance': compliant,
            'input_tokens': input_tokens,
            'avg_output_tokens': avg_output_tokens,
            'token_efficiency_ratio': token_eff_ratio,
            'support_positive': metrics['support']['positive'],
            'support_negative': metrics['support']['negative'],
            'support_neutral': metrics['support']['neutral'],
            'confusion_matrix': metrics['confusion_matrix'],
            'labels': metrics['labels']
        }
        results.append(row)
    return pd.DataFrame(results)

def statistical_tests(df, model, dataset):
    """Perform McNemar tests for techniques within a model-dataset."""

    # Skip McNemar for fp66 and fp100
    if dataset in ['fp66', 'fp100']:
        return []

    sub = df[(df['model'] == model) & (df['dataset'] == dataset)]
    techniques = sub['technique'].unique()
    if len(techniques) < 2:
        return []
    
    pivot = sub.pivot_table(index='statement', columns='technique', values='predicted_sentiment', aggfunc='first')
    true_vals = sub.groupby('statement')['true_sentiment'].first()
    
    correct_matrix = pd.DataFrame(index=pivot.index)
    for tech in techniques:
        correct_matrix[tech] = (pivot[tech] == true_vals).astype(int)
    
    pairwise_results = []
    for tech_a, tech_b in combinations(techniques, 2):
        a = ((correct_matrix[tech_a] == 1) & (correct_matrix[tech_b] == 1)).sum()
        b = ((correct_matrix[tech_a] == 1) & (correct_matrix[tech_b] == 0)).sum()
        c = ((correct_matrix[tech_a] == 0) & (correct_matrix[tech_b] == 1)).sum()
        d = ((correct_matrix[tech_a] == 0) & (correct_matrix[tech_b] == 0)).sum()
        table = np.array([[a, b], [c, d]])
        try:
            result = mcnemar(table, exact=False, correction=True)
            p = result.pvalue
        except:
            p = np.nan
        pairwise_results.append({
            'technique_a': tech_a,
            'technique_b': tech_b,
            'p_value': p,
            'significant': p < 0.05 if not np.isnan(p) else False
        })
    return pairwise_results

def model_comparison_tests(df, dataset, technique):
    """
    Perform McNemar's test comparing Fin-R1 vs Llama for a specific technique on a specific dataset.
    """

    # Skip model comparison for fp66 and fp100 (only Llama data available)
    if dataset in ['fp66', 'fp100']:
        return None

    # Filter for the specific dataset and technique
    sub = df[(df['dataset'] == dataset) & (df['technique'] == technique)]
    
    # Get predictions for each model
    fin_r1 = sub[sub['model'] == 'Fin-R1_GGUF']
    llama = sub[sub['model'] == 'llama3.2-abliterate']
    
    # Ensure we have data for both models
    if fin_r1.empty or llama.empty:
        return None
    
    # Merge on the statement to align paired observations
    merged = pd.merge(
        fin_r1[['statement', 'true_sentiment', 'predicted_sentiment']],
        llama[['statement', 'predicted_sentiment']],
        on='statement',
        suffixes=('_fin', '_llama')
    )
    
    # Calculate correctness
    fin_correct = (merged['predicted_sentiment_fin'] == merged['true_sentiment'])
    llama_correct = (merged['predicted_sentiment_llama'] == merged['true_sentiment'])
    
    # Build contingency table for McNemar
    # a: both correct, b: fin correct, llama wrong
    # c: fin wrong, llama correct, d: both wrong
    a = ((fin_correct == True) & (llama_correct == True)).sum()
    b = ((fin_correct == True) & (llama_correct == False)).sum()
    c = ((fin_correct == False) & (llama_correct == True)).sum()
    d = ((fin_correct == False) & (llama_correct == False)).sum()
    
    table = np.array([[a, b], [c, d]])
    
    try:
        result = mcnemar(table, exact=False, correction=True)
        return {
            'dataset': dataset,
            'technique': technique,
            'fin_correct': fin_correct.sum(),
            'llama_correct': llama_correct.sum(),
            'a_both_correct': a,
            'b_fin_only': b,
            'c_llama_only': c,
            'd_both_wrong': d,
            'p_value': result.pvalue,
            'significant': result.pvalue < 0.05
        }
    except:
        return None

def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_accuracy_comparison(agg_df, group_by_col, output_dir):
    """Plot accuracy comparison bar charts."""

    # For technique level
    if group_by_col == 'technique':
        # For technique_agg, we already have technique as the main column
        # show accuracy for each technique grouped by model/dataset
        # bar chart of accuracy per technique
        plt.figure(figsize=(10, 6))
        avg_accuracy = agg_df.groupby('technique')['accuracy'].mean().sort_values(ascending=False)
        avg_accuracy.plot(kind='bar', color=['#4ECDC4', '#45B7D1', '#FFA502', '#FF6B6B'])
        plt.title('Average Accuracy by Technique')
        plt.ylabel('Accuracy')
        plt.xlabel('Technique')
        plt.ylim(0, 1)
        for i, v in enumerate(avg_accuracy):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_{group_by_col}.png'))
        plt.close()
        return
    
    # For dataset and model levels
    # Aggregate by group_by_col and technique, taking mean accuracy
    agg_plot_df = agg_df.groupby([group_by_col, 'technique'])['accuracy'].mean().reset_index()
    
    # Pivot: rows = group_by_col values, columns = techniques
    pivot_df = agg_plot_df.pivot(index=group_by_col, columns='technique', values='accuracy')
    
    if pivot_df.empty:
        print(f"Warning: No data for {group_by_col} pivot")
        return
    
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title(f'Accuracy Comparison by {group_by_col.capitalize()}')
    plt.ylabel('Accuracy')
    plt.xlabel(group_by_col.capitalize())
    plt.legend(title='Technique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_{group_by_col}.png'))
    plt.close()

def plot_model_performance_per_dataset(df, output_dir):
    """
    Plot model performance (accuracy) per dataset, grouped by technique.
    """
    # Compute accuracy for each model-dataset-technique combination
    results = []
    
    for (model, dataset, technique), group in df.groupby(['model', 'dataset', 'technique']):
        metrics = compute_metrics(group)
        results.append({
            'model': model,
            'dataset': dataset,
            'technique': technique,
            'accuracy': metrics['accuracy']
        })
    
    agg_df = pd.DataFrame(results)
    
    # Pivot: rows = dataset, columns = model, values = accuracy (averaged over techniques)
    pivot_df = agg_df.pivot_table(index='dataset', columns='model', values='accuracy', aggfunc='mean')
    
    if pivot_df.empty:
        print("Warning: No data for model performance per dataset plot")
        return
    
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind='bar', ax=plt.gca())
    plt.title('Model Performance by Dataset (Average Accuracy Across Techniques)')
    plt.ylabel('Accuracy')
    plt.xlabel('Dataset')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_per_dataset.png'))
    plt.close()
    
    print("  Saved model_performance_per_dataset.png")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate prompt engineering experiment results')
    parser.add_argument('--input_dir', required=True, help='Directory containing result CSV files')
    parser.add_argument('--output_dir', default='evaluation_output', help='Output directory for results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all data
    print("Loading result files...")
    df = load_all_results(args.input_dir)
    print(f"Loaded {len(df)} samples from {df['model'].nunique()} models, {df['dataset'].nunique()} datasets, {df['technique'].nunique()} techniques.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ============================================================
    # LEVEL 1: Test Level
    # ============================================================
    print("Computing test-level metrics...")
    test_agg = aggregate_by(df, ['model', 'dataset', 'technique'])
    test_agg.to_csv(os.path.join(args.output_dir, f'test_level_metrics_{timestamp}.csv'), index=False)
    
    # ============================================================
    # LEVEL 2: Dataset Level
    # ============================================================
    print("Computing dataset-level metrics...")
    dataset_agg = aggregate_by(df, ['dataset', 'technique'])
    dataset_agg.to_csv(os.path.join(args.output_dir, f'dataset_level_metrics_{timestamp}.csv'), index=False)
    
    dataset_overall = aggregate_by(df, ['dataset'])
    dataset_overall.to_csv(os.path.join(args.output_dir, f'dataset_overall_metrics_{timestamp}.csv'), index=False)
    
    # ============================================================
    # LEVEL 3: Model Level
    # ============================================================
    print("Computing model-level metrics...")

    # FILTER BEFORE AGGREGATION: Exclude fp66 and fp100
    df_model_filtered = df[~df['dataset'].isin(['fp66', 'fp100'])]

    model_agg = aggregate_by(df, ['model', 'technique'])
    model_agg.to_csv(os.path.join(args.output_dir, f'model_level_metrics_{timestamp}.csv'), index=False)
    
    model_overall = aggregate_by(df, ['model'])
    model_overall.to_csv(os.path.join(args.output_dir, f'model_overall_metrics_{timestamp}.csv'), index=False)
    
    # ============================================================
    # LEVEL 4: Technique Level
    # ============================================================
    print("Computing technique-level metrics...")
    technique_agg = aggregate_by(df, ['technique'])
    technique_agg.to_csv(os.path.join(args.output_dir, f'technique_level_metrics_{timestamp}.csv'), index=False)
    
    # ============================================================
    # OVERALL
    # ============================================================
    print("Computing overall metrics...")
    overall_metrics = compute_metrics(df)
    overall_input_tokens = np.nan
    overall_avg_output_tokens = df['raw_response'].apply(count_tokens).mean()
    overall_token_eff_ratio = np.nan
    overall_row = {
        'accuracy': overall_metrics['accuracy'],
        'macro_f1': overall_metrics['macro_f1'],
        'weighted_f1': overall_metrics['weighted_f1'],
        'precision_positive': overall_metrics['precision']['positive'],
        'precision_negative': overall_metrics['precision']['negative'],
        'precision_neutral': overall_metrics['precision']['neutral'],
        'recall_positive': overall_metrics['recall']['positive'],
        'recall_negative': overall_metrics['recall']['negative'],
        'recall_neutral': overall_metrics['recall']['neutral'],
        'f1_positive': overall_metrics['f1']['positive'],
        'f1_negative': overall_metrics['f1']['negative'],
        'f1_neutral': overall_metrics['f1']['neutral'],
        'support_positive': overall_metrics['support']['positive'],
        'support_negative': overall_metrics['support']['negative'],
        'support_neutral': overall_metrics['support']['neutral'],
        'avg_output_tokens': overall_avg_output_tokens,
    }
    overall_df = pd.DataFrame([overall_row])
    overall_df.to_csv(os.path.join(args.output_dir, f'overall_metrics_{timestamp}.csv'), index=False)
    
    # ============================================================
    # STATISTICAL TESTS: Prompting technique McNemar's test
    # ============================================================
    print("Performing statistical tests...")
    all_tests = []
    for model in df['model'].unique():
        for dataset in df['dataset'].unique():
            tests = statistical_tests(df, model, dataset)
            for pair in tests:
                all_tests.append({
                    'model': model,
                    'dataset': dataset,
                    'technique_a': pair['technique_a'],
                    'technique_b': pair['technique_b'],
                    'p_value': pair['p_value'],
                    'significant': pair['significant']
                })
    stats_df = pd.DataFrame(all_tests)
    if not stats_df.empty:
        stats_df.to_csv(os.path.join(args.output_dir, f'statistical_tests_{timestamp}.csv'), index=False)
    else:
        print("No statistical tests performed (insufficient data).")

    # ============================================================
    # STATISTICAL TESTS: MODEL COMPARISON McNemar's test
    # ============================================================
    print("\nRunning model comparison tests (Fin-R1 vs Llama)...")
    model_test_results = []
    
    for dataset in df['dataset'].unique():
        for technique in df['technique'].unique():
            result = model_comparison_tests(df, dataset, technique)
            if result:
                model_test_results.append(result)
    
    if model_test_results:
        model_test_df = pd.DataFrame(model_test_results)
        model_test_df.to_csv(os.path.join(args.output_dir, f'model_comparison_tests_{timestamp}.csv'), index=False)
        print(f"Model comparison tests saved.")
        
    else:
        print("No model comparison tests performed.")
    
    # ============================================================
    # CONFUSION MATRICES
    # ============================================================
    print("Generating confusion matrices...")
    cm_dir = os.path.join(args.output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    for (model, dataset, technique), group in df.groupby(['model', 'dataset', 'technique']):
        if len(group) == 0:
            continue
        metrics = compute_metrics(group)
        cm = metrics['confusion_matrix']
        labels = metrics['labels']
        title = f"{model}_{dataset}_{technique}"
        plot_confusion_matrix(cm, labels, title, os.path.join(cm_dir, f'cm_{title}.png'))
    
    for dataset, group in df.groupby(['dataset']):
        metrics = compute_metrics(group)
        cm = metrics['confusion_matrix']
        labels = metrics['labels']
        title = f"Dataset_{dataset}"
        plot_confusion_matrix(cm, labels, title, os.path.join(cm_dir, f'cm_dataset_{dataset}.png'))
    
    for model, group in df.groupby(['model']):
        if model == 'llama3.2-abliterate':
            group = group[~group['dataset'].isin(['fp66', 'fp100'])]
        if not group.empty:
            metrics = compute_metrics(group)
            cm = metrics['confusion_matrix']
            labels = metrics['labels']
            title = f"Model_{model}"
            plot_confusion_matrix(cm, labels, title, os.path.join(cm_dir, f'cm_model_{model}.png'))
    
    for technique, group in df.groupby(['technique']):
        metrics = compute_metrics(group)
        cm = metrics['confusion_matrix']
        labels = metrics['labels']
        title = f"Technique_{technique}"
        plot_confusion_matrix(cm, labels, title, os.path.join(cm_dir, f'cm_technique_{technique}.png'))
    
    metrics = compute_metrics(df)
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    plot_confusion_matrix(cm, labels, "Overall", os.path.join(cm_dir, 'cm_overall.png'))
    
    # ============================================================
    # ACCURACY BAR CHARTS
    # ============================================================
    print("Generating accuracy bar charts...")
    
    # Dataset level
    plot_accuracy_comparison(dataset_agg, 'dataset', args.output_dir)
    
    # Model level     
    plot_accuracy_comparison(model_agg, 'model', args.output_dir)
    
    # Technique level
    plot_accuracy_comparison(technique_agg, 'technique', args.output_dir)

    # NEW: Model performance per dataset
    print("Generating model performance per dataset graph...")
    plot_model_performance_per_dataset(df, args.output_dir)
    
    # ============================================================
    # GENERATE REPORT
    # ============================================================
    print("Generating report...")
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EVALUATION REPORT - Prompt Engineering Experiments")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80)
    
    report_lines.append("\nOVERALL PERFORMANCE")
    report_lines.append("-"*40)
    report_lines.append(f"Accuracy: {overall_row['accuracy']:.4f}")
    report_lines.append(f"Macro F1: {overall_row['macro_f1']:.4f}")
    report_lines.append(f"Weighted F1: {overall_row['weighted_f1']:.4f}")
    report_lines.append("\nPer-class metrics:")
    for label in ['positive', 'negative', 'neutral']:
        report_lines.append(f"  {label.capitalize()}: Precision={overall_row[f'precision_{label}']:.4f}, Recall={overall_row[f'recall_{label}']:.4f}, F1={overall_row[f'f1_{label}']:.4f}")
        report_lines.append("\nTOKEN EFFICIENCY (Input & Output)")
        report_lines.append("-"*40)
        report_lines.append(f"Average Output Tokens (overall): {overall_avg_output_tokens:.1f}")
        report_lines.append("\nPer-Technique Token Efficiency:")
        for tech in technique_agg['technique'].unique():
            subset = technique_agg[technique_agg['technique'] == tech]
            if not subset.empty:
                row = subset.iloc[0]
                report_lines.append(f"  {tech}: Input={row['input_tokens']:.1f} tokens, Output={row['avg_output_tokens']:.1f} tokens, Ratio={row['token_efficiency_ratio']:.1f}%")
    
    report_lines.append("\nMODEL-LEVEL PERFORMANCE")
    report_lines.append("-"*40)
    for _, row in model_overall.iterrows():
        report_lines.append(f"\nModel: {row['group']}")
        report_lines.append(f"  Accuracy: {row['accuracy']:.4f}")
        report_lines.append(f"  Macro F1: {row['macro_f1']:.4f}")
    
    report_lines.append("\nDATASET-LEVEL PERFORMANCE")
    report_lines.append("-"*40)
    for _, row in dataset_overall.iterrows():
        report_lines.append(f"\nDataset: {row['group']}")
        report_lines.append(f"  Accuracy: {row['accuracy']:.4f}")
        report_lines.append(f"  Macro F1: {row['macro_f1']:.4f}")
    
    report_lines.append("\nTECHNIQUE-LEVEL PERFORMANCE")
    report_lines.append("-"*40)
    for _, row in technique_agg.iterrows():
        report_lines.append(f"\nTechnique: {row['group']}")
        report_lines.append(f"  Accuracy: {row['accuracy']:.4f}")
        report_lines.append(f"  Macro F1: {row['macro_f1']:.4f}")
    
    if not stats_df.empty:
        report_lines.append("\nSTATISTICAL SIGNIFICANCE TESTS (McNemar's)")
        report_lines.append("-"*40)
        for _, row in stats_df.iterrows():
            sig = "YES" if row['significant'] else "NO"
            report_lines.append(f"{row['model']} | {row['dataset']} | {row['technique_a']} vs {row['technique_b']} | p={row['p_value']:.4f} | Significant: {sig}")
    
    report_path = os.path.join(args.output_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✅ Evaluation complete! Results saved to {args.output_dir}")
    print(f"Report: {report_path}")
    print("Confusion matrices and plots are also saved.")

if __name__ == "__main__":
    main()