"""
QA Verifier
Two-pass verification in a single batched call per batch of QA pairs.
Checks both faithfulness (answer grounded in passage) and answerability
(question can be answered from passage alone).
"""

from tqdm import tqdm

import config
from src.models import GeneratedQA, VerificationResponse, QAVerification
from src.llm_client import GeminiClient


VERIFICATION_PROMPT = """You are a strict quality assurance agent for a financial document QA benchmark dataset. Your job is to verify that generated question-answer pairs are correct and well-grounded.

## YOUR TASK

For each QA pair below, evaluate TWO criteria and assign a confidence score:

### 1. FAITHFULNESS
Is every factual claim in the Answer directly and explicitly supported by the Source Passage?
- Every number in the answer MUST be present in the source passage OR correctly derivable from numbers in the passage (show the derivation in your reasoning).
- No claims should require information beyond what is in the source passage.
- The answer must not add, invent, or hallucinate any facts.

### 2. ANSWERABILITY
Given ONLY the Source Passage and the Question (pretend you haven't seen the answer), could a knowledgeable reader answer this question completely and unambiguously?
- The source passage must contain sufficient information to fully answer the question.
- The question should not require external knowledge, other sections of the document, or unstated context.

### 3. CONFIDENCE SCORE
Assign an overall confidence score from 0.0 to 1.0:
- 1.0 = Perfect. Answer is fully grounded, question is clear, source passage is sufficient.
- 0.8-0.9 = High confidence. Minor stylistic issues but factually correct.
- 0.5-0.7 = Moderate. Some concerns about completeness or clarity.
- 0.0-0.4 = Low. Significant issues with faithfulness or answerability.

## VERDICT RULES
- Set is_faithful=true ONLY if the answer is 100% supported by the source passage.
- Set is_answerable=true ONLY if the passage alone is sufficient to answer the question.
- Be STRICT. When in doubt, mark as false.
- Provide brief reasoning for your decision.

## QA PAIRS TO VERIFY

{qa_pairs_text}

Verify each QA pair above. Return results in order (qa_index 0, 1, 2, ...)."""


def _format_qa_batch(qa_pairs: list[tuple[GeneratedQA, str]], start_index: int = 0) -> str:
    """Format a batch of QA pairs for the verification prompt."""
    parts = []
    for i, (qa, section) in enumerate(qa_pairs):
        parts.append(f"""--- QA Pair {i} ---
Section: {section}
Question: {qa.question}
Answer: {qa.answer}
Source Passage: {qa.source_passage}
Question Type: {qa.question_type.value}
""")
    return "\n".join(parts)


def verify_qa_pairs(
    qa_pairs: list[tuple[GeneratedQA, str]],
    client: GeminiClient,
) -> list[tuple[GeneratedQA, str, QAVerification]]:
    """
    Verify QA pairs in batches using two-pass verification.
    
    Args:
        qa_pairs: List of (GeneratedQA, section_name) tuples
        client: Gemini client instance
        
    Returns:
        List of (GeneratedQA, section_name, QAVerification) for pairs that PASSED
    """
    verified_pairs = []
    rejected_count = 0
    error_count = 0
    batch_size = config.VERIFICATION_BATCH_SIZE
    
    print(f"\n{'='*60}")
    print(f"STAGE: Verification ({len(qa_pairs)} QA pairs in batches of {batch_size})")
    print(f"{'='*60}")
    
    # Process in batches
    batches = [
        qa_pairs[i:i + batch_size]
        for i in range(0, len(qa_pairs), batch_size)
    ]
    
    for batch in tqdm(batches, desc="Verifying", unit="batch"):
        qa_text = _format_qa_batch(batch)
        prompt = VERIFICATION_PROMPT.format(qa_pairs_text=qa_text)
        
        try:
            response = client.generate(
                prompt=prompt,
                response_schema=VerificationResponse,
                temperature=config.VERIFICATION_TEMPERATURE,
            )
            
            # Match verification results to QA pairs
            for result in response.results:
                idx = result.qa_index
                if idx < len(batch):
                    qa, section = batch[idx]
                    if result.is_faithful and result.is_answerable:
                        verified_pairs.append((qa, section, result))
                    else:
                        rejected_count += 1
                        reasons = []
                        if not result.is_faithful:
                            reasons.append("unfaithful")
                        if not result.is_answerable:
                            reasons.append("unanswerable")
                        tqdm.write(f"    ✗ Rejected ({', '.join(reasons)}): {qa.question[:60]}...")
            
        except Exception as e:
            error_count += 1
            tqdm.write(f"    ⚠ Verification error: {str(e)[:80]}")
            # On error, be conservative: skip the batch
            continue
    
    acceptance_rate = len(verified_pairs) / len(qa_pairs) * 100 if qa_pairs else 0
    
    print(f"\n  Verification Results:")
    print(f"    ✓ Accepted: {len(verified_pairs)}")
    print(f"    ✗ Rejected: {rejected_count}")
    print(f"    ⚠ Errors:   {error_count} batches")
    print(f"    Acceptance rate: {acceptance_rate:.1f}%")
    
    return verified_pairs
