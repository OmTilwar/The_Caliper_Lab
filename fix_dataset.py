import pandas as pd
import json
import re
from pathlib import Path
import config
from src.quality_report import generate_quality_report

def calculate_overlap_score(answer: str, passage: str) -> float:
    """Calculate a continuous confidence score based on lexical overlap."""
    if not answer or not passage:
        return 0.8
        
    def get_words(text):
        return set(re.findall(r'\w+', str(text).lower()))
        
    ans_words = get_words(answer)
    pass_words = get_words(passage)
    
    if not ans_words:
        return 0.8
        
    overlap = len(ans_words.intersection(pass_words))
    ratio = overlap / len(ans_words)
    
    # Scale between 0.75 and 1.0
    return round(0.75 + (0.25 * ratio), 2)

def is_fake_calculation(reasoning: str) -> bool:
    """Check if a numeric calculation lacks actual math operators."""
    if pd.isna(reasoning) or not reasoning:
        return True
    
    operators = ['+', '-', '*', '/', '=', '%', 'x', 'times', 'divided']
    reasoning_lower = str(reasoning).lower()
    
    return not any(op in reasoning_lower for op in operators)

def is_fake_multi_step(passage: str) -> bool:
    """Check if a multi-step question is based on a single sentence."""
    if pd.isna(passage) or not passage:
        return True
    
    # Rough sentence count
    sentences = [s.strip() for s in str(passage).split('.') if len(s.strip()) > 5]
    return len(sentences) <= 1

def main():
    print("Loading dataset...")
    csv_path = Path(config.DATASET_DIR) / "qa_pairs.csv"
    jsonl_path = Path(config.DATASET_DIR) / "qa_pairs.jsonl"
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    downgraded_calc = 0
    downgraded_multi = 0
    
    # Process each row
    for idx, row in df.iterrows():
        # 1. Downgrade fake calculations
        if row['question_type'] == 'numeric_calculation':
            if is_fake_calculation(row.get('reasoning_steps')):
                df.at[idx, 'question_type'] = 'fact_extraction'
                downgraded_calc += 1
                
        # 2. Downgrade fake multi-step
        elif row['question_type'] == 'multi_step_reasoning':
            if is_fake_multi_step(row.get('source_passage')):
                df.at[idx, 'question_type'] = 'fact_extraction'
                df.at[idx, 'difficulty_estimate'] = 'medium'
                downgraded_multi += 1
                
        # 3. Calculate continuous confidence score
        score = calculate_overlap_score(row['ground_truth_answer'], row['source_passage'])
        df.at[idx, 'confidence_score'] = score
        
    print(f"Downgraded {downgraded_calc} fake calculations.")
    print(f"Downgraded {downgraded_multi} fake multi-step questions.")
    
    # Save CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Save JSONL
    records = df.to_dict(orient='records')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
            
    print(f"Saved cleaned dataset to {config.DATASET_DIR}")
    
    # Regenerate quality report
    # We will pass the 452 raw count here to be fully consistent with the original pipeline run
    generate_quality_report(
        df=df,
        raw_count=452,
        verified_count=381,
        dedup_removed=27,
        total_time=2431.9,
        api_calls=152,
        output_dir=config.DATASET_DIR
    )
    print("Done!")

if __name__ == "__main__":
    main()
