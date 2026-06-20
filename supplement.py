"""
Supplemental QA Generation
Generates QA pairs for sections that were missed due to API quota limits,
then merges with existing data and re-verifies everything with confidence scores.

Usage:
    python supplement.py
"""

import json
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
from src.models import GeneratedQA, QAVerification, VerifiedQA
from src.quality_report import generate_quality_report


def main():
    load_dotenv()
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print("SUPPLEMENT: Fill Missing Sections + Add Confidence Scores")
    print(f"{'='*60}")
    
    # ── Load existing generation checkpoint ───────────────────
    gen_checkpoint_path = Path(config.CHECKPOINT_DIR) / "generation.json"
    existing_qa_pairs = []
    existing_sections = set()
    
    if gen_checkpoint_path.exists():
        data = json.loads(gen_checkpoint_path.read_text(encoding="utf-8"))
        for item in data:
            qa = GeneratedQA.model_validate(item[0])
            section = item[1]
            existing_qa_pairs.append((qa, section))
            existing_sections.add(section)
        print(f"  Loaded {len(existing_qa_pairs)} existing QA pairs")
        print(f"  Existing sections: {existing_sections}")
    else:
        print("  No generation checkpoint found, starting fresh")
    
    # ── Parse and chunk the document ──────────────────────────
    filepath = download_10k(config.TARGET_COMPANY_CIK, force=False)
    sections = parse_10k(filepath)
    chunks = chunk_sections(sections)
    
    # ── Filter to only chunks from MISSING sections ───────────
    missing_chunks = [c for c in chunks if c.section_name not in existing_sections]
    
    # If all sections are covered but we just need more pairs, 
    # take some chunks from underrepresented sections
    if not missing_chunks:
        print("  All sections already have QA pairs!")
        missing_chunks = []
    else:
        print(f"\n  Missing sections to generate:")
        section_counts = {}
        for c in missing_chunks:
            section_counts[c.section_name] = section_counts.get(c.section_name, 0) + 1
        for name, count in section_counts.items():
            print(f"    {name}: {count} chunks")
        
        # Limit to avoid burning the entire quota
        # Prioritize: MD&A (most important), then Financial Statements (top 5), 
        # then Risk Factors, then Market Risk
        priority_order = [
            "Item 7: MD&A",
            "Item 8: Financial Statements", 
            "Item 1A: Risk Factors",
            "Item 7A: Market Risk Disclosures",
        ]
        
        prioritized = []
        for section_name in priority_order:
            section_chunks = [c for c in missing_chunks if c.section_name == section_name]
            if section_name == "Item 8: Financial Statements":
                # Take first 5 chunks (most content-rich tables)
                prioritized.extend(section_chunks[:5])
            else:
                prioritized.extend(section_chunks)
        
        missing_chunks = prioritized
        print(f"\n  Will generate from {len(missing_chunks)} chunks (prioritized)")
    
    # ── Generate QAs for missing sections ─────────────────────
    client = GeminiClient()
    
    if missing_chunks:
        new_qa_pairs = generate_qa_pairs(missing_chunks, client)
        print(f"  Generated {len(new_qa_pairs)} new QA pairs")
    else:
        new_qa_pairs = []
    
    # ── Merge with existing QA pairs ──────────────────────────
    all_qa_pairs = existing_qa_pairs + new_qa_pairs
    print(f"\n  Total QA pairs (merged): {len(all_qa_pairs)}")
    
    # Save merged generation checkpoint
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    serializable = []
    for qa, section in all_qa_pairs:
        serializable.append([qa.model_dump(), section])
    
    merged_path = checkpoint_dir / "generation.json"
    merged_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    print(f"  Saved merged checkpoint: {merged_path}")
    
    # ── Re-verify ALL pairs with confidence scores ────────────
    print(f"\n  Re-verifying all {len(all_qa_pairs)} pairs with confidence scores...")
    verified_pairs = verify_qa_pairs(all_qa_pairs, client)
    
    # Save verification checkpoint
    ver_serializable = []
    for qa, section, verif in verified_pairs:
        ver_serializable.append([qa.model_dump(), section, verif.model_dump()])
    
    ver_path = checkpoint_dir / "verification.json"
    ver_path.write_text(json.dumps(ver_serializable, indent=2, default=str), encoding="utf-8")
    
    # ── Deduplicate ───────────────────────────────────────────
    deduped_pairs = deduplicate_qa_pairs(verified_pairs)
    
    # ── Output Dataset ────────────────────────────────────────
    output_dir = Path(config.DATASET_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for qa, section, verification in deduped_pairs:
        row = VerifiedQA(
            question=qa.question,
            ground_truth_answer=qa.answer,
            source_passage=qa.source_passage,
            question_type=qa.question_type.value,
            difficulty_estimate=qa.difficulty.value,
            section=section,
            confidence_score=getattr(verification, 'confidence', 1.0),
            reasoning_steps=qa.reasoning_steps,
        )
        rows.append(row)
    
    df = pd.DataFrame([r.model_dump() for r in rows])
    csv_path = output_dir / "qa_pairs.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  CSV saved: {csv_path} ({len(df)} rows)")
    
    jsonl_path = output_dir / "qa_pairs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.model_dump(), ensure_ascii=False) + "\n")
    print(f"  JSONL saved: {jsonl_path} ({len(rows)} rows)")
    
    # ── Quality Report ────────────────────────────────────────
    total_time = time.time() - start_time
    
    raw_count = len(all_qa_pairs)
    verified_count = len(verified_pairs)
    dedup_removed = verified_count - len(deduped_pairs)
    
    generate_quality_report(
        df=df,
        raw_count=raw_count,
        verified_count=verified_count,
        dedup_removed=dedup_removed,
        total_time=total_time,
        api_calls=client.total_requests,
    )
    
    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUPPLEMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  Final dataset: {len(df)} verified QA pairs")
    print(f"  API calls: {client.total_requests}")
    print(f"  Runtime: {total_time:.0f}s ({total_time/60:.1f} min)")
    
    print(f"\n  Section distribution:")
    for section, count in df["section"].value_counts().items():
        print(f"    {section}: {count}")
    
    print(f"\n  Question type distribution:")
    for qt, count in df["question_type"].value_counts().items():
        print(f"    {qt}: {count}")
    
    if len(df) >= 100:
        print(f"\n  TARGET MET: {len(df)} >= 100 pairs")
    else:
        print(f"\n  WARNING: Only {len(df)} pairs (need 100)")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
