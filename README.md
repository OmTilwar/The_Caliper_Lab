# Caliper Lab — 10-K QA Generation Pipeline

An automated pipeline that ingests SEC 10-K filings and generates large, verified question-answer datasets for benchmarking AI models on real financial document tasks.

## Quick Start

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

The pipeline will download NVIDIA's FY2025 10-K, parse it, generate 100+ QA pairs, verify them, and output the dataset to `dataset/`.

### CLI Options

```bash
python pipeline.py --cik 0000320193     # Use a different company (e.g., Apple)
python pipeline.py --min-pairs 150      # Request more QA pairs
python pipeline.py --resume             # Resume from last checkpoint
python pipeline.py --skip-download      # Use cached 10-K file
```

### Evaluation Harness

To score an AI model's answers against the generated dataset:

```bash
# Evaluate a CSV of predictions against the ground truth dataset
python evaluate.py --predictions predictions.csv --dataset dataset/qa_pairs.csv
```
Calculates Exact Match, Token F1, and Numeric Match scores, with a weighted composite score that penalizes hallucinated numbers in calculation questions.

---

## Approach

### Pipeline Architecture

```
[SEC EDGAR] → [Parse & Chunk] → [QA Generation] → [Verification] → [Dedup] → [Dataset]
     │              │                  │                 │              │          │
  Download     BeautifulSoup    OpenRouter/Gemini  OpenRouter/Gemini  TF-IDF    CSV/JSONL
  10-K HTML    + regex section  1 call/chunk       5 QAs/batch       cosine     100+ rows
               detection        all 4 Q types      faithfulness      similarity
                                                   + answerability
                                                   + confidence score
```

### Stage 1: Parse & Chunk
- Downloads the 10-K HTML from SEC EDGAR with proper `User-Agent` header compliance
- Parses the HTML using `BeautifulSoup` with regex-based section detection
- Targets the 5 most content-rich sections: Business, Risk Factors, MD&A, Market Risk, Financial Statements
- **Converts HTML tables to markdown format** to preserve numeric data (critical for calculation questions)
- Splits sections into ~2000-token sub-chunks with 200-token overlap at sentence boundaries

### Stage 2: QA Generation
- **1 API call per chunk** — generates all 4 question types in a single batched prompt
- The prompt attempts to enforce minimum quotas: at least 1 fact_extraction, 1 numeric_calculation, 1 comparison, and 1 multi_step_reasoning question per chunk
- Includes 4 few-shot examples (one per type) to guide quality and format
- Uses native structured output (`response_schema`) with Pydantic models to significantly reduce JSON parsing failures
- Attempts to extract verbatim source passages (character-for-character from the input)
- Requires step-by-step reasoning for numeric and multi-step questions

### Stage 3: Verification
- **Batched verification: 5 QA pairs per API call** — dramatically reduces API usage
- Combined two-check verification in each call:
  - **Faithfulness**: Is every claim in the answer directly supported by the source passage? Are all numbers present or correctly derivable?
  - **Answerability**: Can the question be answered completely using only the source passage?
- **Confidence Score**: The verifier assigns a 0.0 to 1.0 confidence score based on the clarity and grounding of the QA pair.
- A QA pair is accepted **only if both boolean checks pass**
- Temperature set to 0.0 for deterministic, strict evaluation

### Stage 4: Deduplication & Output
- TF-IDF cosine similarity on question strings (scikit-learn, no API calls)
- Threshold: 0.85 — questions above this similarity are flagged as duplicates
- Keeps the higher-difficulty version of duplicate pairs
- Outputs both CSV and JSONL to `dataset/`

---

## Design Choices

### Why a Multi-Backend LLM Strategy?
To handle strict free-tier rate limits, the pipeline supports multiple backends:
- **OpenRouter (Primary)**: Using the `openai` SDK + `instructor`, the pipeline accesses top-tier models (like `gpt-4o-mini` or free options like `llama-3`) which natively support JSON tool-calling and have much higher request limits.
- **Gemini (Fallback)**: Using `google-genai` and Gemini 2.5 Flash as a completely free fallback with native `response_schema` support.
The pipeline automatically routes to OpenRouter if an `OPENROUTER_API_KEY` is present.

### Why Batched Calls?
The free tier limits us to ~250 requests/day. Naive approach needs ~340 calls (4 generation passes + 2 verification passes per QA). Batching reduces this to **~49 calls** — 80% under the limit with room for retries.

### Why Type-Specific Prompt Requirements?
Without explicit quotas, LLMs generate ~80% fact extraction questions (the easiest type). Our prompt mandates a minimum of 1 question per type per chunk, with few-shot examples for each type, ensuring genuine diversity.

### Why Two-Check Verification?
A single "is this correct?" check has a known YES-bias. Separating faithfulness (is the answer grounded?) from answerability (is the passage sufficient?) catches more failure modes:
- Faithfulness catches hallucinated facts and wrong numbers
- Answerability catches questions that require external context

### Why TF-IDF Dedup (Not Embeddings)?
Overlapping chunks produce near-identical questions. TF-IDF cosine similarity is fast, local (no API calls), and effective for detecting text reuse. Embedding APIs would add cost and complexity without meaningful quality improvement for this use case.

---

## Known Limitations

1. **Question Taxonomy Reliability**: Some `multi_step_reasoning` or `hard`-labeled questions are single-sentence restatements with artificial step numbering rather than genuine multi-hop inference across passages. Similarly, some `numeric_calculation` questions extract stated numbers rather than deriving them. The type/difficulty taxonomy needs stronger prompt constraints or a post-hoc classifier to enforce.

2. **Confidence Score Utility**: Confidence scores cluster near 1.0 for all accepted pairs. The verification stage currently behaves more like a pass/fail gate than a graded signal.

3. **Section Imbalance**: The Risk Factors and Market Risk sections are underrepresented (1.4% each) because those sections in this specific 10-K filing are extremely brief and dense with hedged, forward-looking language that's harder to extract clean fact/numeric questions from. This would require section-specific prompting.

4. **Table parsing**: Complex nested HTML tables (common in financial statements) may lose formatting or column alignment during markdown conversion. This can affect the quality of numeric questions generated from tabular data.

5. **Section detection**: Regex-based section detection works well for standard 10-K formats but may miss sections in filings with non-standard HTML structure. The fallback splits the document into equal parts.

6. **Free-tier rate limits**: If using the free Gemini tier, the pipeline takes ~10 minutes due to pacing.

7. **Numeric verification**: The verifier checks if numbers are present in the source passage but cannot independently recompute complex calculations. CoT reasoning steps help but aren't guaranteed correct.

8. **Manual Reporting Process Gap**: The static `project_summary_report.md` contains hardcoded metrics that are not auto-regenerated from the dataset on every pipeline or cleanup script run. The dataset and the report can fall out of sync if the report is not manually re-verified against the CSV before submission.

---

## Scaling Strategy

### Scaling to Multiple Documents (100+ 10-Ks)

1. **Parallel Document Processing**: Use `asyncio` with semaphore-bounded concurrency to process multiple filings simultaneously. Each filing is independent — no inter-document dependencies.

2. **Document Queue**: Deploy a task queue (Celery + Redis) to manage a backlog of CIK numbers. Workers pull filings, run the pipeline, and push results to a shared datastore.

3. **Paid Tier / Batch API**: Gemini's Batch API (available on paid tier) allows submitting thousands of requests at once with 50% cost reduction and much higher rate limits. This alone would scale to 10,000+ QA pairs per day.

### Scaling to 1,000+ QA Pairs

4. **Cross-Document Dedup**: Replace TF-IDF with a vector database (ChromaDB or Pinecone) to detect duplicates across documents, not just within a single filing.

5. **Cost Management**:
   - Prompt caching (Gemini supports this for repeated system prompts)
   - Smaller model for verification (Flash-Lite or a fine-tuned classifier)
   - Batch processing during off-peak hours for lower per-token costs

6. **Quality at Scale**:
   - Human-in-the-loop sampling: randomly audit 5% of generated pairs
   - Automated quality metrics: track acceptance rate, type distribution, difficulty distribution per document
   - Progressive generation: if a document yields <80% acceptance rate, flag for manual review

7. **Rate Limit Handling**:
   - Token bucket algorithm for smooth request distribution
   - Multi-project API key rotation (Gemini free tier is per-project)
   - Automatic failover to backup provider (Groq, OpenRouter) if primary hits limits

---

## Output Format

The dataset is saved in both CSV and JSONL format in `dataset/`:

| Column | Description | Example |
|:---|:---|:---|
| `question` | The generated question | "What was NVIDIA's total revenue in FY2025?" |
| `ground_truth_answer` | Precise answer with specifics | "Total revenue was $130.5 billion..." |
| `source_passage` | Verbatim text from the 10-K | "Total revenue for the fiscal year..." |
| `question_type` | One of 4 types | `numeric_calculation` |
| `difficulty_estimate` | easy / medium / hard | `medium` |
| `section` | Source section in the 10-K | "Item 7: MD&A" |
| `confidence_score` | 0.0 to 1.0 verifier score | `1.0` |
| `reasoning_steps` | CoT steps (for calc/reasoning) | "Step 1: Revenue = $130,497M..." |

---

## Project Structure

```
The_Caliper_Lab/
├── pipeline.py          # Main entry point
├── config.py            # All tunable parameters
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── src/
│   ├── downloader.py    # SEC EDGAR download
│   ├── parser.py        # HTML → sections
│   ├── chunker.py       # Sections → chunks
│   ├── generator.py     # QA generation (Gemini)
│   ├── verifier.py      # QA verification (Gemini)
│   ├── deduplicator.py  # TF-IDF dedup
│   ├── llm_client.py    # Gemini client
│   └── models.py        # Pydantic models
├── dataset/             # Output (CSV + JSONL)
└── data/                # Cached downloads
```

## License

MIT
