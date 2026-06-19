"""
Text Chunker
Splits parsed sections into overlapping sub-chunks for QA generation.
Uses sentence-boundary-aware splitting to preserve context.
"""

import re
from typing import Optional

import nltk

import config
from src.models import Section, Chunk


def _ensure_nltk_data():
    """Download NLTK sentence tokenizer data if not present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("  Downloading NLTK tokenizer data...")
        nltk.download("punkt_tab", quiet=True)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK."""
    _ensure_nltk_data()
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on period + space
        sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences


def chunk_sections(sections: list[Section]) -> list[Chunk]:
    """
    Split sections into overlapping sub-chunks.
    
    Strategy:
    - Split each section into sentences
    - Build chunks of ~CHUNK_SIZE_TOKENS with CHUNK_OVERLAP_TOKENS overlap
    - Respect sentence boundaries (never split mid-sentence)
    - Discard chunks smaller than MIN_CHUNK_SIZE_TOKENS
    
    Args:
        sections: List of parsed sections
        
    Returns:
        List of Chunk objects ready for QA generation
    """
    all_chunks = []
    
    for section in sections:
        sentences = _split_into_sentences(section.full_text)
        if not sentences:
            continue
        
        chunk_index = 0
        current_sentences = []
        current_tokens = 0
        start_char = 0
        
        # Track character positions
        char_pos = 0
        sentence_positions = []
        for sent in sentences:
            pos = section.full_text.find(sent, char_pos)
            if pos == -1:
                pos = char_pos
            sentence_positions.append(pos)
            char_pos = pos + len(sent)
        
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_tokens = _estimate_tokens(sent)
            
            if current_tokens + sent_tokens > config.CHUNK_SIZE_TOKENS and current_sentences:
                # Current chunk is full — save it
                chunk_text = " ".join(current_sentences)
                
                if _estimate_tokens(chunk_text) >= config.MIN_CHUNK_SIZE_TOKENS:
                    end_char = sentence_positions[i - 1] + len(sentences[i - 1]) if i > 0 else len(chunk_text)
                    
                    all_chunks.append(Chunk(
                        section_name=section.name,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        start_char=start_char,
                        end_char=end_char,
                    ))
                    chunk_index += 1
                
                # Calculate overlap: go back by CHUNK_OVERLAP_TOKENS worth of sentences
                overlap_tokens = 0
                overlap_start = len(current_sentences)
                for j in range(len(current_sentences) - 1, -1, -1):
                    overlap_tokens += _estimate_tokens(current_sentences[j])
                    if overlap_tokens >= config.CHUNK_OVERLAP_TOKENS:
                        overlap_start = j
                        break
                
                # Start new chunk from overlap point
                current_sentences = current_sentences[overlap_start:]
                current_tokens = sum(_estimate_tokens(s) for s in current_sentences)
                start_char = sentence_positions[i - len(current_sentences)] if current_sentences else sentence_positions[i]
            
            current_sentences.append(sent)
            current_tokens += sent_tokens
            i += 1
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            if _estimate_tokens(chunk_text) >= config.MIN_CHUNK_SIZE_TOKENS:
                all_chunks.append(Chunk(
                    section_name=section.name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=len(section.full_text),
                ))
    
    print(f"  Created {len(all_chunks)} chunks from {len(sections)} sections")
    for section_name in set(c.section_name for c in all_chunks):
        section_chunks = [c for c in all_chunks if c.section_name == section_name]
        avg_tokens = sum(_estimate_tokens(c.text) for c in section_chunks) // len(section_chunks)
        print(f"    {section_name}: {len(section_chunks)} chunks, ~{avg_tokens} tokens avg")
    
    return all_chunks
