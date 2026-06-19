"""
QA Deduplicator
Removes near-duplicate questions that arise from overlapping chunks.
Uses TF-IDF cosine similarity — no external API calls needed.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import config
from src.models import GeneratedQA, QAVerification


# Difficulty ranking for choosing which duplicate to keep
DIFFICULTY_RANK = {"hard": 3, "medium": 2, "easy": 1}


def deduplicate_qa_pairs(
    qa_pairs: list[tuple[GeneratedQA, str, QAVerification]],
) -> list[tuple[GeneratedQA, str, QAVerification]]:
    """
    Remove near-duplicate QA pairs based on question text similarity.
    When duplicates are found, keeps the higher-difficulty version.
    
    Args:
        qa_pairs: List of verified (GeneratedQA, section_name, QAVerification) tuples
        
    Returns:
        Deduplicated list
    """
    if len(qa_pairs) <= 1:
        return qa_pairs
    
    print(f"\n{'='*60}")
    print(f"STAGE: Deduplication ({len(qa_pairs)} pairs)")
    print(f"{'='*60}")
    
    # Extract question texts for comparison
    questions = [qa.question for qa, _, _ in qa_pairs]
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(questions)
    except ValueError:
        # If all questions are identical or vectorizer fails
        print("  ⚠ TF-IDF vectorization failed, skipping dedup")
        return qa_pairs
    
    # Compute pairwise cosine similarity
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # Find duplicate pairs (above threshold)
    to_remove = set()
    duplicate_count = 0
    
    for i in range(len(qa_pairs)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(qa_pairs)):
            if j in to_remove:
                continue
            
            if sim_matrix[i, j] > config.SIMILARITY_THRESHOLD:
                duplicate_count += 1
                qa_i = qa_pairs[i][0]
                qa_j = qa_pairs[j][0]
                
                # Keep the higher-difficulty version
                rank_i = DIFFICULTY_RANK.get(qa_i.difficulty.value, 0)
                rank_j = DIFFICULTY_RANK.get(qa_j.difficulty.value, 0)
                
                if rank_i >= rank_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
    
    # Filter out duplicates
    deduped = [
        qa_pairs[i] for i in range(len(qa_pairs)) if i not in to_remove
    ]
    
    removed = len(qa_pairs) - len(deduped)
    print(f"  Found {duplicate_count} duplicate pairs")
    print(f"  Removed {removed} duplicates, {len(deduped)} remaining")
    
    return deduped
