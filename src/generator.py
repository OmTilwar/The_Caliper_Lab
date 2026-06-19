"""
QA Generator
Generates question-answer pairs from document chunks using Gemini.
Uses a single batched call per chunk that produces all 4 question types.
"""

from tqdm import tqdm

import config
from src.models import Chunk, GeneratedQA, GenerationResponse
from src.llm_client import GeminiClient


GENERATION_PROMPT = """You are a senior financial analyst creating a benchmark dataset from a 10-K SEC filing. Your task is to generate high-quality question-answer pairs from the passage below.

## REQUIREMENTS

Generate exactly {num_questions} question-answer pairs from this passage. You MUST include:
- At least 1 "fact_extraction" question (who, what, when, where — direct lookup)
- At least 1 "numeric_calculation" question (requires arithmetic: percentage change, ratios, differences)
- At least 1 "comparison" question (compare across years, segments, categories, or metrics)
- At least 1 "multi_step_reasoning" question (requires combining 2+ facts or logical steps)

## STRICT RULES

1. **source_passage**: Must be the EXACT verbatim text copied character-for-character from the passage below. Do NOT paraphrase, summarize, or modify the source text in any way.
2. **answer**: Must be precise and fully supported by the source_passage. Include specific numbers, dates, and units.
3. **reasoning_steps**: REQUIRED for numeric_calculation and multi_step_reasoning questions. Show step-by-step work.
4. **difficulty**:
   - "easy": Single fact lookup, answer is one entity/number directly stated
   - "medium": Requires understanding context, simple arithmetic, or identifying relationships
   - "hard": Multi-step reasoning, complex calculations, or synthesis of multiple facts

## FEW-SHOT EXAMPLES

**Example 1 — fact_extraction (easy):**
Question: "What was NVIDIA's total revenue for fiscal year 2025?"
Answer: "NVIDIA's total revenue for fiscal year 2025 was $130.5 billion."
Source passage: "Total revenue for the fiscal year ended January 26, 2025 was $130,497 million, compared to $60,922 million for the prior year."
Reasoning steps: null

**Example 2 — numeric_calculation (medium):**
Question: "What was the year-over-year percentage increase in NVIDIA's total revenue from fiscal 2024 to fiscal 2025?"
Answer: "Revenue increased by approximately 114.2% year-over-year, from $60.9 billion to $130.5 billion."
Source passage: "Total revenue for the fiscal year ended January 26, 2025 was $130,497 million, compared to $60,922 million for the prior year."
Reasoning steps: "Step 1: Revenue FY2025 = $130,497M. Step 2: Revenue FY2024 = $60,922M. Step 3: Change = $130,497M - $60,922M = $69,575M. Step 4: Percentage change = ($69,575M / $60,922M) × 100 = 114.2%."

**Example 3 — comparison (medium):**
Question: "How did Data Center revenue compare to Gaming revenue in fiscal 2025?"
Answer: "Data Center revenue of $115.2 billion was significantly larger than Gaming revenue of $11.4 billion, representing approximately 10.1x the Gaming segment."
Source passage: "Data Center revenue was $115,199 million and Gaming revenue was $11,359 million for fiscal year 2025."
Reasoning steps: "Data Center = $115,199M, Gaming = $11,359M. Ratio: $115,199M / $11,359M ≈ 10.1x."

**Example 4 — multi_step_reasoning (hard):**
Question: "If NVIDIA's Data Center segment maintained its fiscal 2025 revenue share, what would its revenue be given a hypothetical total revenue of $200 billion?"
Answer: "Data Center's share was approximately 88.3% of total revenue. At $200 billion total, Data Center revenue would be approximately $176.6 billion."
Source passage: "Total revenue for the fiscal year ended January 26, 2025 was $130,497 million. Data Center revenue was $115,199 million."
Reasoning steps: "Step 1: Data Center share = $115,199M / $130,497M = 88.28%. Step 2: At $200B total: $200B × 0.8828 = $176.6B."

## PASSAGE FROM 10-K FILING (Section: {section_name})

{chunk_text}

Generate {num_questions} diverse, high-quality QA pairs following the requirements above."""


def generate_qa_pairs(
    chunks: list[Chunk],
    client: GeminiClient,
) -> list[tuple[GeneratedQA, str]]:
    """
    Generate QA pairs from all chunks.
    Makes 1 API call per chunk with all 4 question types.
    
    Args:
        chunks: List of text chunks to process
        client: Gemini client instance
        
    Returns:
        List of (GeneratedQA, section_name) tuples
    """
    all_qa_pairs = []
    
    print(f"\n{'='*60}")
    print(f"STAGE: QA Generation ({len(chunks)} chunks)")
    print(f"{'='*60}")
    
    for chunk in tqdm(chunks, desc="Generating QAs", unit="chunk"):
        # Determine number of questions based on chunk size
        chunk_tokens = len(chunk.text) // 4
        if chunk_tokens > 1500:
            num_q = config.QA_PER_CHUNK_MAX
        elif chunk_tokens > 800:
            num_q = config.QA_PER_CHUNK_MIN + 1
        else:
            num_q = config.QA_PER_CHUNK_MIN
        
        prompt = GENERATION_PROMPT.format(
            num_questions=num_q,
            section_name=chunk.section_name,
            chunk_text=chunk.text,
        )
        
        try:
            response = client.generate(
                prompt=prompt,
                response_schema=GenerationResponse,
                temperature=config.GENERATION_TEMPERATURE,
            )
            
            for qa in response.qa_pairs:
                all_qa_pairs.append((qa, chunk.section_name))
            
            tqdm.write(f"    ✓ {chunk.section_name} [chunk {chunk.chunk_index}]: {len(response.qa_pairs)} QAs")
            
        except Exception as e:
            tqdm.write(f"    ✗ {chunk.section_name} [chunk {chunk.chunk_index}]: Error - {str(e)[:80]}")
            continue
    
    print(f"\n  Total raw QA pairs generated: {len(all_qa_pairs)}")
    
    # Print distribution
    type_counts = {}
    diff_counts = {}
    for qa, _ in all_qa_pairs:
        type_counts[qa.question_type.value] = type_counts.get(qa.question_type.value, 0) + 1
        diff_counts[qa.difficulty.value] = diff_counts.get(qa.difficulty.value, 0) + 1
    
    print("  By type:", dict(type_counts))
    print("  By difficulty:", dict(diff_counts))
    
    return all_qa_pairs
