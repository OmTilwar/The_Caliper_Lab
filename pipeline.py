"""
Caliper Lab — 10-K QA Generation Pipeline
Main orchestrator that runs all pipeline stages.

Usage:
    python pipeline.py                          # Full run with defaults (NVIDIA)
    python pipeline.py --cik 0000320193         # Use Apple
    python pipeline.py --min-pairs 150          # Generate more pairs
    python pipeline.py --resume                 # Resume from last checkpoint
    python pipeline.py --skip-download          # Use cached 10-K
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import config
from src.downloader import download_10k
from src.parser import parse_10k
from src.chunker import chunk_sections
from src.generator import generate_qa_pairs
from src.verifier import verify_qa_pairs
from src.deduplicator import deduplicate_qa_pairs
from src.llm_client import GeminiClient
from src.models import VerifiedQA


def save_checkpoint(data: list, stage: str):
    """Save intermediate results to checkpoint file."""
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / f"{stage}.json"
    
    # Convert to serializable format
    serializable = []
    for item in data:
        if isinstance(item, tuple):
            serializable.append([
                obj.model_dump() if hasattr(obj, 'model_dump') else obj
                for obj in item
            ])
        elif hasattr(item, 'model_dump'):
            serializable.append(item.model_dump())
        else:
            serializable.append(item)
    
    filepath.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    print(f"  💾 Checkpoint saved: {filepath}")


def load_checkpoint(stage: str):
    """Load checkpoint if it exists."""
    filepath = Path(config.CHECKPOINT_DIR) / f"{stage}.json"
    if filepath.exists():
        data = json.loads(filepath.read_text(encoding="utf-8"))
        print(f"  📂 Loaded checkpoint: {filepath}")
        return data
    return None


def output_dataset(
    verified_pairs: list[tuple],
    output_dir: str = None,
):
    """
    Write the final dataset to CSV and JSONL.
    
    Args:
        verified_pairs: List of (GeneratedQA, section_name, QAVerification) tuples
    """
    output_dir = Path(output_dir or config.DATASET_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to VerifiedQA objects
    rows = []
    for qa, section, verification in verified_pairs:
        row = VerifiedQA(
            question=qa.question,
            ground_truth_answer=qa.answer,
            source_passage=qa.source_passage,
            question_type=qa.question_type.value,
            difficulty_estimate=qa.difficulty.value,
            section=section,
            reasoning_steps=qa.reasoning_steps,
        )
        rows.append(row)
    
    # Write CSV
    df = pd.DataFrame([r.model_dump() for r in rows])
    csv_path = output_dir / "qa_pairs.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  📄 CSV saved: {csv_path} ({len(df)} rows)")
    
    # Write JSONL
    jsonl_path = output_dir / "qa_pairs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
    print(f"  📄 JSONL saved: {jsonl_path} ({len(rows)} rows)")
    
    return df


def print_summary(df: pd.DataFrame, total_time: float, api_calls: int):
    """Print final pipeline summary statistics."""
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  📊 Dataset Statistics:")
    print(f"     Total verified QA pairs: {len(df)}")
    print(f"     Runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"     API calls used: {api_calls}")
    
    print(f"\n  📋 Question Type Distribution:")
    for qt, count in df["question_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"     {qt}: {count} ({pct:.1f}%)")
    
    print(f"\n  📋 Difficulty Distribution:")
    for diff, count in df["difficulty_estimate"].value_counts().items():
        pct = count / len(df) * 100
        print(f"     {diff}: {count} ({pct:.1f}%)")
    
    print(f"\n  📋 Section Distribution:")
    for section, count in df["section"].value_counts().items():
        pct = count / len(df) * 100
        print(f"     {section}: {count} ({pct:.1f}%)")
    
    print(f"\n  ✅ Dataset files saved to: {config.DATASET_DIR}/")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Caliper Lab — 10-K QA Generation Pipeline"
    )
    parser.add_argument(
        "--cik", default=config.TARGET_COMPANY_CIK,
        help=f"SEC CIK number (default: {config.TARGET_COMPANY_CIK})"
    )
    parser.add_argument(
        "--min-pairs", type=int, default=config.MIN_QA_PAIRS,
        help=f"Minimum QA pairs to generate (default: {config.MIN_QA_PAIRS})"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use cached 10-K"
    )
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"CALIPER LAB — 10-K QA GENERATION PIPELINE")
    print(f"{'='*60}")
    print(f"  Company CIK: {args.cik}")
    print(f"  Target QA pairs: ≥{args.min_pairs}")
    print(f"  Model: {config.GEMINI_MODEL}")
    print(f"  Rate limit: {config.GEMINI_RPM} RPM")
    print(f"{'='*60}")
    
    # ── Stage 1: Download ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 1: Download 10-K")
    print(f"{'='*60}")
    
    filepath = download_10k(args.cik, force=False)
    
    # ── Stage 2: Parse ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 2: Parse 10-K HTML")
    print(f"{'='*60}")
    
    sections = parse_10k(filepath)
    
    # ── Stage 3: Chunk ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 3: Chunk Sections")
    print(f"{'='*60}")
    
    chunks = chunk_sections(sections)
    
    # ── Initialize Gemini Client ──────────────────────────────
    client = GeminiClient()
    
    # ── Stage 4: Generate QA Pairs ────────────────────────────
    raw_qa_pairs = None
    
    if args.resume:
        checkpoint = load_checkpoint("generation")
        if checkpoint:
            from src.models import GeneratedQA, QuestionType, Difficulty
            raw_qa_pairs = []
            for item in checkpoint:
                qa_data = item[0]
                section = item[1]
                qa = GeneratedQA.model_validate(qa_data)
                raw_qa_pairs.append((qa, section))
    
    if raw_qa_pairs is None:
        raw_qa_pairs = generate_qa_pairs(chunks, client)
        save_checkpoint(raw_qa_pairs, "generation")
    
    # ── Stage 5: Verify QA Pairs ──────────────────────────────
    verified_pairs = None
    
    if args.resume:
        checkpoint = load_checkpoint("verification")
        if checkpoint:
            from src.models import GeneratedQA, QAVerification
            verified_pairs = []
            for item in checkpoint:
                qa = GeneratedQA.model_validate(item[0])
                section = item[1]
                verif = QAVerification.model_validate(item[2])
                verified_pairs.append((qa, section, verif))
    
    if verified_pairs is None:
        verified_pairs = verify_qa_pairs(raw_qa_pairs, client)
        save_checkpoint(verified_pairs, "verification")
    
    # ── Stage 6: Deduplicate ──────────────────────────────────
    deduped_pairs = deduplicate_qa_pairs(verified_pairs)
    
    # ── Check if we have enough pairs ─────────────────────────
    if len(deduped_pairs) < args.min_pairs:
        print(f"\n  ⚠ Only {len(deduped_pairs)} pairs after dedup (need {args.min_pairs})")
        print(f"  Proceeding with what we have...")
    
    # ── Stage 7: Output Dataset ───────────────────────────────
    print(f"\n{'='*60}")
    print("STAGE 7: Output Dataset")
    print(f"{'='*60}")
    
    df = output_dataset(deduped_pairs)
    
    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - start_time
    print_summary(df, total_time, client.total_requests)


if __name__ == "__main__":
    main()
