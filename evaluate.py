"""
Evaluation Harness
Scores a model's predicted answers against the ground truth dataset.
Demonstrates that the evaluation logic actually works.

Usage:
    python evaluate.py --predictions predictions.csv
    python evaluate.py --predictions predictions.csv --dataset dataset/qa_pairs.csv
"""

import argparse
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation/articles/whitespace."""
    text = str(text).lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


def extract_numbers(text: str) -> list[float]:
    """Extract all numbers from a string for numeric comparison."""
    # Match integers, decimals, percentages, currency amounts
    pattern = r'[\$]?\d+[,\d]*\.?\d*[%]?'
    matches = re.findall(pattern, str(text))
    numbers = []
    for m in matches:
        clean = m.replace('$', '').replace('%', '').replace(',', '')
        try:
            numbers.append(float(clean))
        except ValueError:
            continue
    return numbers


def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary exact match after normalization. Returns 0.0 or 1.0."""
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score between prediction and ground truth.
    Standard SQuAD-style evaluation metric.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def numeric_match(prediction: str, ground_truth: str, tolerance: float = 0.02) -> float:
    """
    Check if key numbers in the ground truth appear in the prediction.
    Uses relative tolerance for comparison.
    Returns the fraction of ground truth numbers found in prediction.
    """
    gt_numbers = extract_numbers(ground_truth)
    pred_numbers = extract_numbers(prediction)
    
    if not gt_numbers:
        return 1.0  # No numbers to check
    
    matched = 0
    for gt_num in gt_numbers:
        for pred_num in pred_numbers:
            if gt_num == 0 and pred_num == 0:
                matched += 1
                break
            elif gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) <= tolerance:
                matched += 1
                break
    
    return matched / len(gt_numbers)


def evaluate_single(prediction: str, ground_truth: str, question_type: str) -> dict:
    """Evaluate a single prediction against ground truth."""
    em = exact_match(prediction, ground_truth)
    f1 = token_f1(prediction, ground_truth)
    num = numeric_match(prediction, ground_truth)
    
    # Composite score: weight numeric accuracy higher for calculation questions
    if question_type in ('numeric_calculation', 'multi_step_reasoning'):
        composite = 0.3 * f1 + 0.7 * num
    else:
        composite = 0.7 * f1 + 0.3 * em
    
    return {
        'exact_match': em,
        'token_f1': f1,
        'numeric_match': num,
        'composite_score': composite,
    }


def evaluate_dataset(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict:
    """
    Evaluate a full predictions dataset against ground truth.
    
    Args:
        predictions_df: Must have columns ['question', 'predicted_answer']
        ground_truth_df: The QA pairs dataset with ground truth
        
    Returns:
        Dictionary with overall and per-type metrics
    """
    # Merge on question text
    merged = ground_truth_df.merge(
        predictions_df[['question', 'predicted_answer']],
        on='question',
        how='inner',
    )
    
    if len(merged) == 0:
        print("  ERROR: No matching questions found between predictions and ground truth.")
        print("  Make sure the 'question' column matches exactly.")
        return {}
    
    print(f"  Matched {len(merged)} / {len(ground_truth_df)} questions")
    
    # Evaluate each pair
    results = []
    for _, row in merged.iterrows():
        scores = evaluate_single(
            row['predicted_answer'],
            row['ground_truth_answer'],
            row['question_type'],
        )
        scores['question_type'] = row['question_type']
        scores['difficulty'] = row['difficulty_estimate']
        results.append(scores)
    
    results_df = pd.DataFrame(results)
    
    # Overall metrics
    overall = {
        'total_evaluated': len(results_df),
        'exact_match': results_df['exact_match'].mean(),
        'token_f1': results_df['token_f1'].mean(),
        'numeric_match': results_df['numeric_match'].mean(),
        'composite_score': results_df['composite_score'].mean(),
    }
    
    # Per question type
    by_type = {}
    for qt in results_df['question_type'].unique():
        qt_df = results_df[results_df['question_type'] == qt]
        by_type[qt] = {
            'count': len(qt_df),
            'exact_match': qt_df['exact_match'].mean(),
            'token_f1': qt_df['token_f1'].mean(),
            'numeric_match': qt_df['numeric_match'].mean(),
            'composite_score': qt_df['composite_score'].mean(),
        }
    
    # Per difficulty
    by_difficulty = {}
    for diff in results_df['difficulty'].unique():
        diff_df = results_df[results_df['difficulty'] == diff]
        by_difficulty[diff] = {
            'count': len(diff_df),
            'composite_score': diff_df['composite_score'].mean(),
        }
    
    return {
        'overall': overall,
        'by_question_type': by_type,
        'by_difficulty': by_difficulty,
    }


def print_results(results: dict):
    """Pretty-print evaluation results."""
    if not results:
        return
    
    overall = results['overall']
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n  Overall Metrics ({overall['total_evaluated']} questions):")
    print(f"    Exact Match:     {overall['exact_match']:.3f}")
    print(f"    Token F1:        {overall['token_f1']:.3f}")
    print(f"    Numeric Match:   {overall['numeric_match']:.3f}")
    print(f"    Composite Score: {overall['composite_score']:.3f}")
    
    print(f"\n  By Question Type:")
    for qt, metrics in results['by_question_type'].items():
        print(f"    {qt} (n={metrics['count']}): "
              f"F1={metrics['token_f1']:.3f}, "
              f"Numeric={metrics['numeric_match']:.3f}, "
              f"Composite={metrics['composite_score']:.3f}")
    
    print(f"\n  By Difficulty:")
    for diff, metrics in results['by_difficulty'].items():
        print(f"    {diff} (n={metrics['count']}): "
              f"Composite={metrics['composite_score']:.3f}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against the QA dataset"
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to CSV with columns: question, predicted_answer"
    )
    parser.add_argument(
        "--dataset", default="dataset/qa_pairs.csv",
        help="Path to ground truth dataset CSV"
    )
    args = parser.parse_args()
    
    # Load data
    gt_df = pd.read_csv(args.dataset)
    pred_df = pd.read_csv(args.predictions)
    
    print(f"\n  Ground truth: {len(gt_df)} questions")
    print(f"  Predictions:  {len(pred_df)} answers")
    
    # Evaluate
    results = evaluate_dataset(pred_df, gt_df)
    print_results(results)


if __name__ == "__main__":
    main()
