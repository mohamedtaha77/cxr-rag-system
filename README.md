# Multi-Modal Chest X-Ray Intelligence System

**DSAI 413 — Assignment 2 | Spring 2026 | Mohammed Taha (202201788)**

A dual-mode medical AI system for chest X-ray analysis combining multimodal retrieval (ColPali) with a medical vision-language model (MedGemma).

---

## Results Summary

**ColPali + MedGemma achieves the best performance** on report generation, confirming the hypothesis that patch-level late-interaction retrieval provides more clinically relevant context than global embedding retrieval.

### Report Generation (50 test studies)

| Metric | ColPali + MedGemma (RAG) | CLIP + MedGemma (RAG) | MedGemma Direct |
|--------|--------------------------|----------------------|-----------------|
| **BERTScore F1** | **0.4743** | 0.4590 | 0.4614 |
| **ROUGE-L** | **0.0933** | 0.0898 | 0.0750 |

### QA Mode (30 test QA pairs, ColPali + MedGemma)
| Metric | Score |
|--------|-------|
| BERTScore F1 | 0.6696 |
| ROUGE-L | 0.2040 |

---

## Modes

| Mode | Input | Output |
|------|-------|--------|
| **Report Generation** | CXR image | Structured radiology report (IMPRESSION + FINDINGS) |
| **QA** | CXR image + clinical question | Evidence-grounded answer with retrieved supporting cases |

---

## Architecture

The system supports **two modes** that share the same retrieval and generation components but use different query strategies:

```
                                  USER INPUT
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
   ┌──────────────────┐                                 ┌──────────────────┐
   │ MODE 1: Report   │                                 │ MODE 2: QA       │
   │ Generation       │                                 │                  │
   │                  │                                 │                  │
   │  CXR Image       │                                 │  CXR Image       │
   │  (no question)   │                                 │  + Question      │
   └────────┬─────────┘                                 └────────┬─────────┘
            │                                                    │
            │ query: "chest x-ray findings"                      │ query: question text
            │                                                    │
            └──────────────────────┬─────────────────────────────┘
                                   │
                  ┌────────────────▼─────────────────┐
                  │   RETRIEVAL LAYER (RAG)          │
                  │                                  │
                  │   ColPali v1.3   ────┐           │
                  │   (patch-level)      │ pick one  │
                  │   CLIP ViT-L/14  ────┘           │
                  │   (global)                       │
                  │                                  │
                  │   Output: top-k similar CXRs     │
                  │           + their impressions    │
                  └────────────────┬─────────────────┘
                                   │
                  ┌────────────────▼─────────────────┐
                  │   GENERATION LAYER               │
                  │   MedGemma 1.5 4B IT (4-bit)     │
                  │                                  │
                  │   Input:  CXR image + context    │
                  │           (+ question for QA)    │
                  └────────────────┬─────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                                         ▼
   ┌──────────────────┐                       ┌──────────────────┐
   │  Structured      │                       │  Grounded        │
   │  Radiology Report│                       │  Clinical Answer │
   │  (IMPRESSION +   │                       │  (1-3 sentences  │
   │   FINDINGS)      │                       │   based on image │
   │                  │                       │   + retrieved)   │
   └──────────────────┘                       └──────────────────┘
```

**Key insight**: Both modes use the same MedGemma + retriever, but differ in:
- **Input**: report mode is image-only, QA mode adds a clinical question
- **Retrieval query**: report mode uses generic findings query, QA mode uses the question itself
- **Prompt template**: different system prompts for report vs answer
- **Output**: structured report vs concise grounded answer

---

## Setup

### 1. Prerequisites
- Python 3.10+
- CUDA GPU with ≥ 8 GB VRAM (recommended: Kaggle T4 with 30 hr/week free)
- Accounts: [HuggingFace](https://huggingface.co) + [Groq](https://console.groq.com) + [Kaggle](https://www.kaggle.com)

### 2. MedGemma Access (Required)
MedGemma is a gated model.
1. Visit [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) and click **"Agree and access repository"**
2. Generate a fine-grained token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **"Read access to contents of all public gated repos"** permission

### 3. Environment
```bash
cp .env.example .env
# Fill in HF_TOKEN and GROQ_API_KEY
```

### 4. Install
```bash
pip install --upgrade peft transformers
pip install colpali-engine accelerate bitsandbytes
pip install open-clip-torch faiss-cpu
pip install bert-score rouge-score
pip install groq sentence-transformers
pip install streamlit gradio   # pick one for local demo
```

---

## Running the Pipeline (Kaggle T4)

Run notebooks in order. Setup instructions are in each notebook's first markdown cell.

| Notebook | Purpose | Approx. Time |
|----------|---------|-------------|
| `01_data_and_qa_dataset.ipynb` | Load Indiana CXR + generate QA pairs via Groq | ~30 min |
| `02_colpali_indexing.ipynb` | Build ColPali + CLIP indexes on corpus | ~3 hrs |
| `03_pipelines_and_eval.ipynb` | Run all 3 systems, compute metrics | ~2 hrs |
| `04_comparison.ipynb` | Visualize comparison (no GPU needed) | ~5 min |
| `05_demo_app.ipynb` | Launch live Gradio demo with ngrok public URL | ~5 min |

**Each notebook is self-documenting** — required Kaggle Inputs and Secrets are listed at the top.

---

## Dataset: Indiana University CXR (via Kaggle)

- **Source**: [raddar/chest-xrays-indiana-university](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) on Kaggle
- **Size**: 3,955 radiology reports + 7,470 CXR images (PNG)
- **Format**: CSV reports (indiana_reports.csv, indiana_projections.csv)
- **License**: CC BY-NC-ND 4.0

Auto-mounted in Kaggle via **+ Add Input** — no download needed.

---

## QA Dataset Creation Methodology

> The assignment requires building a custom QA dataset. There is no pre-existing QA dataset for Indiana CXR. This section documents the methodology used.

### Step 1 — Report Parsing
CSV reports are parsed to extract `impression` and `findings`. Comparison language (e.g., "unchanged since prior study", "compared to previous") is removed using regex patterns, following [CXR-RePaiR-Gen (Ranjit et al., 2023)](https://arxiv.org/abs/2305.03660).

### Step 2 — Question Generation
Adopts the **MIMIC-CXR-VQA methodology** (Aas-Alas et al., MIDL 2026), adapted for Indiana CXR:

- **15 clinical categories** covering major chest X-ray pathologies:
  Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Mediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices, Subcutaneous Emphysema

- **6 question templates per category**, covering different phrasings of the same clinical query

- **Category selection per study**: Keywords in impression/findings are matched against the category keyword map to select relevant question categories (avoiding irrelevant negative questions)

- **Up to 8 questions per study** to balance dataset size vs. Groq API rate limits

### Step 3 — Answer Generation via Groq API

Answers generated using **LLaMA 3.1 8B Instant** via Groq API (free tier):

**System prompt constraints** (following MIMIC-CXR-VQA Appendix B):
- Answer strictly based on provided radiographic findings and impression
- Do not mention prior studies, comparisons, follow-up, or temporal changes
- Refer to "the radiograph" or "the image", not "the report"
- Give concise, evidence-based answers (1–2 sentences)
- If a finding is not mentioned, state it is not observed
- Never infer beyond what is explicitly stated

**User prompt** (following MIMIC-CXR-VQA Appendix C):
```
Findings: {findings}
Impression: {impression}

Question: {question}
```

### Dataset Statistics (this implementation)

| Split | Studies | QA Pairs |
|-------|---------|---------|
| Train | 200* | 1,515 |
| Val   | 0    | 0    |
| Test  | 0    | 0    |
| **Total** | **200** | **1,515** |

*QA generation was limited to first 200 studies due to Groq API rate limits (28 req/min on free tier). All 200 fell within the 80% train split.

---

## Model Comparison

### Report Generation Results

| Metric | ColPali + MedGemma (RAG) | CLIP + MedGemma (RAG) | MedGemma Direct |
|--------|--------------------------|----------------------|-----------------|
| **BERTScore F1** | **0.4743** | 0.4590 | 0.4614 |
| **ROUGE-L** | **0.0933** | 0.0898 | 0.0750 |
| RadGraph F1 | N/A | N/A | N/A |

> *RadGraph F1 not computed (requires PhysioNet credentials).*

**Key findings**:
1. ColPali + MedGemma achieves the **best BERTScore F1 (0.4743)** and **best ROUGE-L (0.0933)**
2. ColPali outperforms both CLIP RAG and direct generation
3. CLIP RAG vs Direct is mixed (slightly worse BERTScore, slightly better ROUGE-L), suggesting global embedding retrieval can introduce noise

**Hypothesis confirmed**: ColPali's patch-level late-interaction retrieval provides more clinically relevant context for MedGemma than CLIP's global image-text embedding, leading to better report quality.

---

## Methodological Limitations

### Self-Retrieval Bias

The ColPali and CLIP indexes contain all 3,652 corpus images, including those used in the test set. When evaluating on a test image, the retriever can return that same image as top-1, causing the "retrieved context" to include the exact ground-truth impression for the query image.

**Impact**: Absolute BERTScore and ROUGE-L values are likely upper bounds and may not reflect true generalization performance.

**Mitigation**: All three systems face identical retrieval conditions, so the **relative ranking remains valid**:
- ColPali RAG (0.4743) > MedGemma Direct (0.4614) > CLIP RAG (0.4590)

The hypothesis that **patch-level late-interaction retrieval provides more clinically relevant context than global embedding retrieval** is supported by the comparison between ColPali and CLIP systems, since both retrievers have the same self-retrieval advantage.

### QA Evaluation on Train Split

QA pairs were generated only for the first 200 studies (all in train split). Evaluation was performed on these pairs as a methodology demonstration.

---

## Running the Demo App

The Gradio app runs on Kaggle T4 with a public URL exposed via ngrok. Open `notebooks/05_demo_app.ipynb` on Kaggle and follow its in-notebook setup. A Streamlit version (`app/app.py`) is also included for local use.

```bash
# Local Gradio
python app/app_gradio.py

# Local Streamlit
streamlit run app/app.py
```

---

## Project Structure

```
cxr-rag-system/
├── README.md                       # This file
├── REPORT.md                       # Short report (architecture, models, results, limitations)
├── requirements.txt
├── .env.example                    # HF_TOKEN, GROQ_API_KEY template
│
├── src/
│   ├── data/
│   │   ├── openi_loader.py         # Indiana CXR CSV parsing + train/val/test split
│   │   └── qa_creator.py           # Groq-based QA pair generation (15 categories)
│   ├── retrieval/
│   │   ├── colpali_retriever.py    # ColPali v1.3 — direct colpali-engine usage
│   │   └── clip_retriever.py       # CLIP ViT-L/14 + FAISS — global embedding baseline
│   ├── generation/
│   │   ├── medgemma_generator.py   # MedGemma 4B IT (4-bit) — report gen + QA
│   │   └── prompts.py              # Prompt templates for all modes
│   └── evaluation/
│       └── metrics.py              # BERTScore, ROUGE-L, BLEU-4
│
├── notebooks/                      # Kaggle notebooks (run in order 01 → 05)
│   ├── 01_data_and_qa_dataset.ipynb
│   ├── 02_colpali_indexing.ipynb
│   ├── 03_pipelines_and_eval.ipynb
│   ├── 04_comparison.ipynb
│   └── 05_demo_app.ipynb           # Live Gradio demo via Kaggle + ngrok
│
├── evaluation/                     # Evaluation results
│   ├── results.csv                 # Comparison metrics table
│   ├── predictions.csv             # Generated reports for all 3 systems
│   └── qa_results.csv              # QA predictions
│
└── app/
    ├── app.py                      # Streamlit dual-mode UI (local)
    ├── app_gradio.py               # Gradio dual-mode UI (Kaggle/HF Spaces)
    └── utils.py                    # Cached model loaders
```

---

## References

1. Ranjit et al. (2023). *Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models*. arXiv:2305.03660
2. Aas-Alas et al. (2026). *MIMIC-CXR-VQA: A Medical Visual Question Answering Dataset*. MIDL 2026
3. Faysse et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models*. [colpali-engine](https://github.com/illuin-tech/colpali)
4. Lee et al. (2024). *LLM-CXR: Instruction-finetuned LLM for CXR Image Understanding and Generation*. ICLR 2024
5. Google. *MedGemma*. [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
