# KK-GPT

KK-GPT is an official-source Retrieval-Augmented Generation (RAG) system for U.S. immigration information.

The V1 scope covers:

* F-1
* CPT
* OPT
* STEM OPT
* H-1B
* H-4
* PERM / ETA Form 9089
* I-140

The system is designed to answer factual immigration questions using retrieved evidence from official U.S. government sources while conservatively abstaining when sufficient evidence is not available.

> KK-GPT provides informational guidance only and is not legal advice.

---

## Architecture

```text
User
 ↓
Streamlit UI
 ↓ HTTP
FastAPI /ask
 ↓
Query Policy
 ├── Non-factual / predictive / recommendation query
 │      ↓
 │   Policy response
 │
 └── OFFICIAL_FACT
        ↓
    Query Understanding
        ↓
    Topic Filtering
        ↓
    Semantic Retrieval
        ↓
    Source Diversity
        ↓
    Source Policy
        ↓
    Evidence Sufficiency Gate
        ↓
    Grounded Generation
        ↓
    Citation Validation
        ↓
    Answer or Abstain
```

---

## Design Principles

KK-GPT V1 separates several problems that are often incorrectly combined in RAG systems.

### Retrieval relevance is not answerability

A highly similar document does not necessarily mean that a question should be answered.

KK-GPT therefore separates:

```text
Query classification
↓
Retrieval
↓
Evidence sufficiency
↓
Generation
```

This prevents the system from treating vector similarity as confidence.

### Conservative abstention

If retrieved official evidence is insufficient, the system returns:

```text
I do not have enough official evidence to answer this question.
```

The pipeline fails closed rather than allowing unsupported generation.

### Official-source grounding

V1 uses official government material such as:

* eCFR
* USCIS
* DHS / SEVP
* Department of Labor material

Community posts and unofficial sources are intentionally excluded from V1.

### Source-aware retrieval

Retrieval includes source-diversity constraints so that the context is not dominated by many chunks from the same document.

The default V1 retrieval path does not use cross-encoder reranking because evaluation showed that reranking reduced retrieval performance on the project's gold dataset.

---

## RAG Pipeline

### 1. Ingestion

The ingestion pipeline processes official immigration material from:

* HTML pages
* PDFs
* eCFR API/XML content

Documents are normalized and converted into chunk-level records with metadata.

An important ingestion failure discovered during development involved eCFR pages returning successful HTTP responses while containing only navigation-shell content.

The ingestion pipeline was changed to use official eCFR API/XML data and semantic content validation.

This demonstrated an important production lesson:

> HTTP success does not guarantee successful document ingestion.

---

## 2. Chunking

Documents are converted into retrieval chunks while preserving metadata such as:

* source ID
* agency
* document title
* topic
* URL
* authority type
* status
* dates

The final V1 corpus contains:

```text
1,371 chunks
```

---

## 3. Embeddings

KK-GPT uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Embeddings are normalized and stored locally.

---

## 4. Query Policy

Before retrieval, each question is classified into one of the following categories:

```text
OFFICIAL_FACT
CASE_PREDICTION
FUTURE_PREDICTION
RECOMMENDATION
PERSONAL_DECISION
LOCAL_SERVICE
UNKNOWN
```

Only:

```text
OFFICIAL_FACT
```

is routed into the factual RAG pipeline.

This prevents questions such as case predictions or personalized legal recommendations from being treated as ordinary retrieval questions.

---

## 5. Retrieval

The frozen V1 retrieval architecture is:

```text
Query normalization
→ Topic filtering
→ Semantic retrieval
→ Source diversity
→ Top-K evidence
```

Configuration:

```text
Retrieval Top-K: 5
Maximum chunks per source: 2
```

Cross-encoder reranking was experimentally evaluated but excluded from the default V1 pipeline after reducing benchmark performance.

---

## 6. Source Intelligence

Sources contain metadata used to represent authority and status.

Authority ordering includes:

```text
REGULATION
POLICY_MANUAL
OFFICIAL_GUIDANCE
FORM_INSTRUCTIONS
ARCHIVED_GUIDANCE
```

The system can also reason about source status such as:

```text
CURRENT
SUPERSEDED
ARCHIVED
WITHDRAWN
```

When authoritative current sources genuinely conflict without a deterministic winner, the architecture is designed to surface the conflict rather than silently selecting one source.

---

## 7. Evidence Sufficiency

Retrieval alone does not authorize an answer.

Retrieved evidence is passed through an evidence-sufficiency gate before generation.

The gate determines whether the available official evidence is sufficient to answer the user's specific question.

If not, the system abstains.

V1 uses up to:

```text
5 evidence chunks
```

for this decision.

---

## 8. Grounded Generation

The local language model is:

```text
qwen3:4b-instruct
```

served through Ollama.

Generation is instructed to:

* use only retrieved official evidence
* avoid outside knowledge
* provide inline citation IDs
* abstain if the evidence cannot support the answer

---

## 9. Citation Validation

Generated citation IDs are validated against the evidence supplied to the model.

This deterministic validation confirms that cited IDs actually exist in the supplied context.

Semantic citation support is evaluated separately during offline evaluation.
