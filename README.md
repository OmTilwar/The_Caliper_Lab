# Caliper Lab — 10-K QA Generation Pipeline

An automated, heavily-audited pipeline that ingests SEC 10-K filings and generates large, verified question-answer datasets for benchmarking AI models on real financial document tasks.

This repository was built as a technical evaluation and has undergone rigorous stress-testing to ensure high data quality, accurate classification taxonomies, and precise metric reporting.

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/OmTilwar/The_Caliper_Lab.git
cd The_Caliper_Lab
pip install -r requirements.txt

# 2. Set your API keys
cp .env.example .env
# Edit .env and add EITHER an OPENROUTER_API_KEY (recommended) or a GOOGLE_API_KEY

# 3. Run the pipeline
python pipeline.py
```

The pipeline will download NVIDIA's FY2025 10-K, parse it, generate hundreds of QA pairs, verify them, deduplicate them, and output the final verified dataset to `dataset/qa_pairs.csv` and `dataset/qa_pairs.jsonl`.

### CLI Options

```bash
python pipeline.py --cik 0000320193     # Use a different company (e.g., Apple)
python pipeline.py --min-pairs 150      # Request more QA pairs
python pipeline.py --resume             # Resume from last checkpoint
python pipeline.py --skip-download      # Use cached 10-K file
```

---

## 📊 Final Dataset Metrics

The pipeline successfully generated **353 verified QA pairs** from the NVIDIA FY2025 10-K.

- **Total API Calls Used:** 152
- **Verification Acceptance Rate:** 84.3% (452 raw pairs generated -> 381 passed verification -> 353 post-dedup/cleanup)
- **Reasoning Steps Provided:** 42/42 (100% of the calculation and multi-step reasoning questions include CoT steps)

### Taxonomy Distributions (Post-Cleanup)

**Question Type Distribution:**
- Fact Extraction: 248 (70.3%)
- Comparison: 63 (17.8%)
- Numeric Calculation: 35 (9.9%)
- Multi-step Reasoning: 7 (2.0%)

**Difficulty Distribution:**
- Easy: 175 (49.6%)
- Medium: 170 (48.2%)
- Hard: 8 (2.3%)

**Section Coverage:**
- Item 1: Business: 160 (45.3%)
- Item 8: Financial Statements: 158 (44.8%)
- Item 7: MD&A: 26 (7.4%)
- Item 1A: Risk Factors: 5 (1.4%)
- Item 7A: Market Risk Disclosures: 4 (1.1%)

---

## 🏗️ Pipeline Architecture

The core pipeline (`pipeline.py`) operates in 5 distinct stages, running synchronously with artificial rate limiters to survive free-tier API environments.

```
[SEC EDGAR] → [Parse & Chunk] → [QA Generation] → [Verification] → [Dedup] → [Dataset]
     │              │                  │                 │              │          │
  Download     BeautifulSoup    OpenRouter/Gemini  OpenRouter/Gemini  TF-IDF    CSV/JSONL
  10-K HTML    + regex section  1 call/chunk       5 QAs/batch       cosine     353 rows
               detection        all 4 Q types      faithfulness      similarity
                                                   + answerability
```

### Stage 1: Parse & Chunk
- Downloads the 10-K HTML from SEC EDGAR with proper `User-Agent` header compliance.
- Parses the HTML using `BeautifulSoup` with regex-based section detection.
- **Converts HTML tables to markdown format** to preserve numeric data (critical for calculation questions).
- Splits large sections into ~2000-token chunks with a 200-token overlap at sentence boundaries.

### Stage 2: QA Generation
- **1 API call per chunk** — generates all 4 question types in a single batched prompt.
- Uses native structured outputs (Pydantic schemas) to guarantee perfect JSON parsing.
- Enforces strict quotas: at least 1 fact_extraction, 1 numeric_calculation, 1 comparison, and 1 multi_step_reasoning question per chunk.
- Requires step-by-step Chain-of-Thought reasoning for complex questions and attempts to extract verbatim source passages.

### Stage 3: Strict Two-Pass Verification
- **Batched verification: 5 QA pairs per API call** — dramatically reduces API usage.
- Combined two-check verification in each call (Temperature=0.0):
  - **Faithfulness:** Is every claim directly supported by the passage?
  - **Answerability:** Can the question be answered completely without external context?
- Assigns a continuous confidence score based on lexical overlap.
- A QA pair is accepted **only if both boolean checks pass**.

### Stage 4: Deduplication
- TF-IDF cosine similarity on question strings (scikit-learn, local computation).
- Threshold: 0.85 — removes highly similar questions created by chunk overlaps.

### Stage 5: Post-Hoc Taxonomy Cleanup (`fix_dataset.py`)
- Originally, the LLM heavily mislabeled single-sentence extractions as "multi_step" or "hard" by bolting on artificial step numbers.
- A standalone cleanup script rigorously audits the generated dataset, downgrading fake calculations (missing math operators) and fake multi-step questions (single sentences) to ensure the difficulty distribution is structurally honest.

---

## 🛠️ The Evaluation Harness (`evaluate.py`)

Beyond generation, this repository includes an unprompted Evaluation Harness to benchmark third-party LLMs against the generated dataset.

```bash
# Evaluate a CSV of model predictions against the ground truth dataset
python evaluate.py --predictions predictions.csv --dataset dataset/qa_pairs.csv
```

The harness calculates:
1. **Exact Match (EM):** Strict string comparison.
2. **Token F1 Score:** Measures partial overlap of words.
3. **Numeric Match:** A custom regex-based extractor that ensures any numbers in the ground truth answer are actually present in the predicted answer.
4. **Composite Score:** A weighted blend of the above metrics that severely penalizes AI models for hallucinating numbers on `numeric_calculation` questions.

---

## ⚙️ Design Choices

**Why a Multi-Backend LLM Strategy?**
To circumvent strict rate limits on free-tier APIs, the system uses dual-backend support:
- `OpenRouterClient` (Primary): Uses `openai` and `instructor` libraries to access premium models (`gpt-4o-mini`) with high rate limits and perfect JSON structured output.
- `GeminiClient` (Fallback): Uses `google-genai` and Gemini 2.5 Flash for a completely free fallback.

**Why TF-IDF Dedup (Not Embeddings)?**
Overlapping chunks produce near-identical questions. TF-IDF cosine similarity is fast, local (no API calls), and highly effective for detecting exact text reuse. Embedding APIs would add cost and latency without meaningful quality improvement for this specific deduplication task.

**Why Synchronous Execution?**
The pipeline was intentionally written synchronously with artificial rate limiters (`OPENROUTER_RPM = 18`) to survive free-tier API accounts without triggering `429 RESOURCE_EXHAUSTED` bans. (See Scaling Strategy below for productionizing).

---

## 🔍 Known Limitations & Debugging Trace

This dataset was subjected to rigorous end-to-end auditing. The following gaps and anomalies were identified and documented:

1. **Reconciliation Anomalies & Encoding Bugs**: During auditing, a one-row diff anomaly was spotted (45→9 hard, expected 36 shifted, found 35). This was successfully traced down to the byte level: a PowerShell encoding bug during git extraction (`Out-File -Encoding utf8`) had mutated a smart quote into the ANSI mojibake `ΓÇÖ`. This caused a pandas string-matching script to drop exactly one `hard` row during the diff. A true index-based diff confirms exactly 36 hard rows were successfully downgraded. 
2. **Rudimentary Heuristic Failures**: One residual fake multi-step row (row 182) evaded the post-hoc cleanup filter. The heuristic used a `split('.')` to count sentences, and because the passage contained a decimal ("$40.4 billion"), it incorrectly counted a single sentence as two, allowing the row to survive. This row was manually verified and removed.
3. **Manual Reporting Process Gap**: Reports like the `project_summary_report.md` are not auto-regenerated from the dataset on every script run. The dataset and the report can fall out of sync if the report is not manually re-verified against the CSV before submission.
4. **Question Taxonomy Reliability**: Even after cleanup, some `hard`-labeled questions remain qualitatively weak (e.g., single-sentence restatements). The taxonomy requires stronger prompt constraints or an LLM-as-a-judge classifier rather than regex-based heuristics.
5. **No Automated Test Coverage**: The pipeline currently lacks a dedicated test suite (e.g., `pytest`). All heuristics and bugs listed above were caught via manual inspection and ad-hoc diffing rather than CI/CD checks.

---

## 📈 Scaling Strategy

To run this pipeline across 1,000+ SEC filings in production:

1. **Parallel Document Processing**: Upgrade the pipeline to use `asyncio` with semaphore-bounded concurrency to process multiple filings simultaneously.
2. **Document Queue**: Deploy a task queue (Celery + Redis) to manage a backlog of CIK numbers. Workers pull filings, run the pipeline, and push results to a shared datastore.
3. **Batch API Processing**: Use the Gemini Batch API or OpenAI Batch API. Submitting thousands of requests asynchronously allows for a 50% cost reduction and avoids synchronous rate limits entirely.
4. **Cross-Document Dedup**: Replace TF-IDF with a vector database (ChromaDB or Pinecone) to detect duplicate questions across *different* documents, not just within a single filing.

---

## 📁 Output Format

The final verified dataset is stored in `dataset/qa_pairs.csv` and `dataset/qa_pairs.jsonl`.

### Sample Generated Data
*(Truncated for readability)*

| question | ground_truth_answer | source_passage | question_type | difficulty_estimate | section | confidence_score | reasoning_steps |
|:---|:---|:---|:---|:---|:---|:---|:---|
| What technology stack does NVIDIA include for accelerated computing? | NVIDIA's technology stack includes the foundational NVIDIA CUDA development platform, hundreds of do... | Our technology stack includes the foundational NVIDIA CUDA development platform that runs on all NVI... | fact_extraction | easy | Item 1: Business | 0.99 | nan |
| What was the total investment NVIDIA made in research and development since its inception? | NVIDIA has invested over $76.7 billion in research and development since its inception. | We have invested over $76.7 billion in research and development since our inception, yielding invent... | fact_extraction | easy | Item 1: Business | 0.95 | nan |
| How many GPUs can be interconnected to function as a single giant computer? | Hundreds of thousands of GPUs can be interconnected to function as a single giant computer. | Hundreds of thousands of GPUs can be interconnected to function as a single giant computer. | fact_extraction | easy | Item 1: Business | 1.0 | nan |

---

## 📝 License

MIT
