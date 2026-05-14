# Multi-Modal Chest X-Ray Intelligence System

**DSAI 413 — Assignment 2 | Spring 2026 | Mohammed Taha (202201788)**

A dual-mode medical AI system for chest X-ray analysis combining multimodal retrieval (ColPali) with a medical vision-language model (MedGemma).

---

## Modes

| Mode | Input | Output |
|------|-------|--------|
| **Report Generation** | CXR image | Structured radiology report (IMPRESSION + FINDINGS) |
| **QA** | CXR image + clinical question | Evidence-grounded answer with retrieved supporting cases |

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              User Input                  │
                    │  CXR Image  +  (optional) Question       │
                    └──────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Retrieval Layer (RAG)      │
                    │                              │
                    │  ColPali v1.3 (Primary)      │  ← late-interaction over
                    │  CLIP ViT-L/14 (Baseline)    │    image patches (ColPali)
                    │                              │    or global embeddings (CLIP)
                    │  → Top-k similar CXR images  │
                    │  → Retrieved impression text │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Generation Layer           │
                    │   MedGemma 1.5 4B IT         │
                    │   (4-bit NF4 quantization)   │
                    │                              │
                    │  Input: image + context      │
                    │  Output: report / answer     │
                    └─────────────────────────────┘
```

---

## Setup

### 1. Prerequisites
- Python 3.10+
- CUDA GPU with ≥ 8 GB VRAM for local use (recommended: Google Colab T4/A100)
- Accounts: [HuggingFace](https://huggingface.co) + [Groq](https://console.groq.com)

### 2. MedGemma Access (Required)
MedGemma is a gated model. Visit [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it), agree to the health AI terms, then generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Environment
```bash
cp .env.example .env
# Fill in HF_TOKEN and GROQ_API_KEY
```

### 4. Install (in Colab — see notebooks for exact order)
```bash
pip install colpali-engine byaldi
pip install "transformers>=4.45.0" accelerate bitsandbytes
pip install open-clip-torch faiss-gpu
pip install bert-score radgraph rouge-score
pip install groq sentence-transformers streamlit
```

---

## Running the Pipeline (Colab)

Run notebooks in order:

| Notebook | Purpose | Approx. Time |
|----------|---------|-------------|
| `01_data_and_qa_dataset.ipynb` | Download OpenI dataset + generate QA pairs via Groq | 30 min – 3 hrs |
| `02_colpali_indexing.ipynb` | Build ColPali + CLIP indexes | 1.5 – 2 hrs |
| `03_pipelines_and_eval.ipynb` | Run all 3 systems, compute metrics | 2 – 3 hrs |
| `04_comparison.ipynb` | Visualize comparison | 20 min |

**All large downloads happen inside Colab's data center — not through your local WiFi.**

---

## Dataset: OpenI (Indiana University CXR)

**No registration required.** The dataset is freely available from the NIH NLM.

- **Size**: ~3,955 radiology reports + 7,470 CXR images (PNG format)
- **Format**: XML reports with structured `IMPRESSION` and `FINDINGS` sections
- **License**: Public domain

Downloaded automatically inside Notebook 01:
```bash
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.zip
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
```

---

## QA Dataset Creation Methodology

> The assignment requires building a custom QA dataset. There is no pre-existing QA dataset for OpenI. This section documents the exact methodology used.

### Step 1 — Report Parsing
XML reports are parsed to extract the `IMPRESSION` and `FINDINGS` sections. Comparison language (e.g., "unchanged since prior study", "compared to previous") is removed using regex patterns, following the approach in [CXR-RePaiR-Gen (Ranjit et al., 2023)](https://arxiv.org/abs/2305.03660).

### Step 2 — Question Generation
We adopt the **MIMIC-CXR-VQA methodology** (Aas-Alas et al., MIDL 2026), adapted for OpenI:

- **15 clinical categories** covering all major chest X-ray pathologies:
  Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Mediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices, Subcutaneous Emphysema

- **6 question templates per category**, covering different phrasings of the same clinical query (e.g., "Is there evidence of X?", "Can X be identified?", "Are there signs of X?")

- **Category selection per study**: Keywords in the impression/findings text are matched against the category keyword map to select relevant question categories for that study (avoiding irrelevant negative questions)

- **Up to 8 questions per study** are generated to balance dataset size vs. Groq API rate limits

### Step 3 — Answer Generation via Groq API

Answers are generated using **LLaMA 3.1 8B Instant** via the Groq API (free tier):

**System prompt constraints** (following MIMIC-CXR-VQA Appendix B):
- Answer strictly based on the provided radiographic findings and impression
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

### Dataset Statistics

| Split | Studies | QA Pairs |
|-------|---------|---------|
| Train | ~3,164 | ~19,000 |
| Val   | ~395   | ~2,370  |
| Test  | ~396   | ~2,376  |
| **Total** | **~3,955** | **~23,746** |

Category distribution and full dataset statistics are generated in Notebook 01.

---

## Model Comparison

| Metric | ColPali + MedGemma (RAG) | CLIP + MedGemma (RAG) | MedGemma Direct |
|--------|--------------------------|----------------------|-----------------|
| BERTScore F1 | — | — | — |
| ROUGE-L | — | — | — |
| RadGraph F1 | — | — | — |
| QA BERTScore F1 | — | — | — |

*(Populated after running Notebooks 03–04)*

**Key hypothesis**: ColPali's patch-level late-interaction retrieval provides more clinically relevant context for MedGemma than CLIP's global image-text embedding, leading to better report quality.

---

## Running the Demo App

### Locally
```bash
streamlit run app/app.py
```

### HuggingFace Spaces
The app is deployed at: `https://huggingface.co/spaces/YOUR_USERNAME/cxr-rag-system`

Set Space Secrets: `HF_TOKEN`, `GROQ_API_KEY`, `INDEX_DIR`, `CORPUS_PATH`

---

## Project Structure

```
cxr-rag-system/
├── src/
│   ├── data/
│   │   ├── openi_loader.py      # OpenI XML parsing + train/val/test split
│   │   └── qa_creator.py        # Groq-based QA pair generation (15 categories)
│   ├── retrieval/
│   │   ├── colpali_retriever.py # ColPali via Byaldi — late-interaction retrieval
│   │   └── clip_retriever.py    # CLIP ViT-L/14 + FAISS — global embedding baseline
│   ├── generation/
│   │   ├── medgemma_generator.py # MedGemma 4B IT (4-bit) — report gen + QA
│   │   └── prompts.py            # Prompt templates for all modes
│   └── evaluation/
│       └── metrics.py            # BERTScore, ROUGE-L, RadGraph F1, BLEU-4
├── notebooks/                    # Colab notebooks (run in order 01 → 04)
├── app/
│   ├── app.py                    # Streamlit dual-mode UI
│   └── utils.py                  # Cached model loaders
└── data/
    ├── processed/
    │   ├── reports_corpus.csv    # Parsed OpenI reports
    │   └── qa_dataset.jsonl      # Generated QA pairs
    └── sample_images/            # Sample CXRs for demo
```

---

## References

1. Ranjit et al. (2023). *Retrieval Augmented Chest X-Ray Report Generation using OpenAI GPT models*. arXiv:2305.03660
2. Aas-Alas et al. (2026). *MIMIC-CXR-VQA: A Medical Visual Question Answering Dataset*. MIDL 2026
3. Faysse et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models*. [colpali-engine](https://github.com/illuin-tech/colpali)
4. Lee et al. (2024). *LLM-CXR: Instruction-finetuned LLM for CXR Image Understanding and Generation*. ICLR 2024
5. Google. *MedGemma*. [huggingface.co/google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
