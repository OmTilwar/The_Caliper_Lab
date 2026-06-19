"""
Pydantic data models for the QA generation pipeline.
Used both as internal data structures and as Gemini response schemas.
"""

from __future__ import annotations

from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────

class QuestionType(str, Enum):
    FACT_EXTRACTION = "fact_extraction"
    NUMERIC_CALCULATION = "numeric_calculation"
    COMPARISON = "comparison"
    MULTI_STEP_REASONING = "multi_step_reasoning"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─── Internal Data Models ─────────────────────────────────────────

class Section(BaseModel):
    """A parsed section from the 10-K filing."""
    name: str = Field(description="Section name, e.g. 'Item 1A: Risk Factors'")
    full_text: str = Field(description="Full text content of the section")


class Chunk(BaseModel):
    """A sub-chunk of a section, ready for QA generation."""
    section_name: str = Field(description="Parent section name")
    chunk_index: int = Field(description="Index of this chunk within the section")
    text: str = Field(description="Text content of the chunk")
    start_char: int = Field(description="Start character offset in the section")
    end_char: int = Field(description="End character offset in the section")


# ─── Gemini Response Schemas ──────────────────────────────────────

class GeneratedQA(BaseModel):
    """A single generated question-answer pair (used in Gemini response schema)."""
    question: str = Field(description="The generated question about the financial document")
    answer: str = Field(description="The precise, correct answer to the question")
    source_passage: str = Field(
        description="The exact verbatim text from the passage that supports the answer. "
                    "Must be copied character-for-character from the input passage."
    )
    question_type: QuestionType = Field(
        description="Classification of the question type"
    )
    difficulty: Difficulty = Field(
        description="Difficulty estimate: easy (single fact lookup), "
                    "medium (contextual understanding or simple math), "
                    "hard (multi-step reasoning or complex calculation)"
    )
    reasoning_steps: Optional[str] = Field(
        default=None,
        description="Step-by-step reasoning or calculation steps. "
                    "Required for numeric_calculation and multi_step_reasoning questions."
    )


class GenerationResponse(BaseModel):
    """Response schema for the QA generation call."""
    qa_pairs: list[GeneratedQA] = Field(
        description="List of generated question-answer pairs"
    )


class QAVerification(BaseModel):
    """Verification result for a single QA pair."""
    qa_index: int = Field(
        description="Zero-based index of the QA pair in the batch being verified"
    )
    is_faithful: bool = Field(
        description="True if every claim in the answer is directly and explicitly "
                    "supported by the source passage. All numbers must be present "
                    "in or correctly derivable from the passage."
    )
    is_answerable: bool = Field(
        description="True if the question can be completely and unambiguously "
                    "answered using only the source passage, without external knowledge."
    )
    reasoning: str = Field(
        description="Brief explanation of the verification decision"
    )


class VerificationResponse(BaseModel):
    """Response schema for the batched verification call."""
    results: list[QAVerification] = Field(
        description="Verification results for each QA pair in the batch"
    )


# ─── Final Output Model ──────────────────────────────────────────

class VerifiedQA(BaseModel):
    """A verified QA pair ready for the final dataset output."""
    question: str
    ground_truth_answer: str
    source_passage: str
    question_type: str
    difficulty_estimate: str
    section: str
    reasoning_steps: Optional[str] = None
