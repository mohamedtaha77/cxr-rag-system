# HuggingFace Spaces Deployment — Step by Step

Estimated time: **~1.5 hours total** if you follow this exactly.

---

## Phase 1: Upload indexes to HF Dataset (~20 min)

The indexes (~1 GB total) are too big for a Space repo. Upload them as a HF Dataset and download at runtime.

### 1.1 Create a HF Dataset
1. Go to https://huggingface.co/new-dataset
2. Owner: your username (`mohamedtaha77`)
3. Dataset name: `cxr-rag-indexes`
4. Visibility: **Private** (or public if you don't mind)
5. Click **Create dataset**

### 1.2 Upload files (use Kaggle notebook for fast upload)
Run this in a Kaggle notebook (where your files already are):

```python
!pip install -q huggingface_hub

from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')
api = HfApi(token=HF_TOKEN)

REPO_ID = 'mohamedtaha77/cxr-rag-indexes'

# Upload files
files_to_upload = {
    '/kaggle/input/notebooks/mohammedtaha778/01-02-data-qa-indexes-complete/colpali_index/colpali_embeddings.pt': 'colpali_index/colpali_embeddings.pt',
    '/kaggle/input/notebooks/mohammedtaha778/01-02-data-qa-indexes-complete/colpali_index/colpali_paths.pkl': 'colpali_index/colpali_paths.pkl',
    '/kaggle/input/notebooks/mohammedtaha778/01-02-data-qa-indexes-complete/clip_index/clip_faiss.index': 'clip_index/clip_faiss.index',
    '/kaggle/input/notebooks/mohammedtaha778/01-02-data-qa-indexes-complete/clip_index/clip_paths.pkl': 'clip_index/clip_paths.pkl',
    '/kaggle/input/notebooks/mohammedtaha778/01-02-data-qa-indexes-complete/reports_corpus.csv': 'reports_corpus.csv',
}

for local_path, repo_path in files_to_upload.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type='dataset',
    )
    print(f'✓ Uploaded {repo_path}')
```

**Adjust the local paths** to match wherever your files are in Kaggle Inputs.

---

## Phase 2: Create HuggingFace Space (~10 min)

### 2.1 Create Space
1. Go to https://huggingface.co/new-space
2. Owner: your username
3. Space name: `cxr-rag-demo`
4. SDK: **Streamlit**
5. Hardware: **ZeroGPU** (free, A10G GPU)
6. Visibility: **Public**
7. Click **Create Space**

### 2.2 Add Secrets
In your Space → **Settings** → **Variables and secrets**:
- Add `HF_TOKEN` (your HuggingFace token)
- Add `GROQ_API_KEY` (optional, not needed for the demo app)
- Add `INDEX_REPO` with value `mohamedtaha77/cxr-rag-indexes`

---

## Phase 3: Adapt the app for HF Spaces (~30 min)

The current app needs these changes:

### 3.1 Replace `app/utils.py` with HF-aware version

```python
"""
Model and index loading utilities for HuggingFace Spaces.
"""
import os
import streamlit as st
from huggingface_hub import hf_hub_download

INDEX_REPO = os.environ.get("INDEX_REPO", "mohamedtaha77/cxr-rag-indexes")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

INDEX_DIR = "/tmp/indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


@st.cache_resource(show_spinner="Downloading indexes from HuggingFace...")
def download_indexes():
    """Download all index files from HF Dataset."""
    files = [
        "colpali_index/colpali_embeddings.pt",
        "colpali_index/colpali_paths.pkl",
        "clip_index/clip_faiss.index",
        "clip_index/clip_paths.pkl",
        "reports_corpus.csv",
    ]
    for f in files:
        hf_hub_download(
            repo_id=INDEX_REPO,
            filename=f,
            repo_type="dataset",
            local_dir=INDEX_DIR,
            token=HF_TOKEN,
        )
    return INDEX_DIR


@st.cache_resource(show_spinner="Loading ColPali...")
def load_colpali():
    download_indexes()
    from src.retrieval.colpali_retriever import ColPaliRetriever
    return ColPaliRetriever.from_index(os.path.join(INDEX_DIR, "colpali_index"))


@st.cache_resource(show_spinner="Loading CLIP...")
def load_clip():
    download_indexes()
    from src.retrieval.clip_retriever import CLIPRetriever
    retriever = CLIPRetriever()
    retriever.load_index(os.path.join(INDEX_DIR, "clip_index"))
    return retriever


@st.cache_resource(show_spinner="Loading MedGemma (4-bit)...")
def load_medgemma():
    from src.generation.medgemma_generator import MedGemmaGenerator
    return MedGemmaGenerator(hf_token=HF_TOKEN, load_in_4bit=True)


@st.cache_data(show_spinner=False)
def load_corpus():
    import pandas as pd
    download_indexes()
    df = pd.read_csv(os.path.join(INDEX_DIR, "reports_corpus.csv"))
    return dict(zip(df["study_id"], df["impression"]))


def get_retriever(choice: str):
    if "ColPali" in choice:
        return load_colpali()
    return load_clip()


def study_id_from_path(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]
```

### 3.2 Add `requirements.txt` for the Space

```
streamlit
torch
transformers
peft
accelerate
bitsandbytes
colpali-engine
open-clip-torch
faiss-cpu
sentence-transformers
huggingface_hub
pandas
pillow
```

### 3.3 Add `spaces` decorator to `app.py` (optional but recommended)

Wrap the generation calls with `@spaces.GPU` decorator for ZeroGPU allocation:

```python
import spaces

@spaces.GPU(duration=120)
def generate_with_gpu(generator, image, context_reports=None, structured=False):
    return generator.generate_report(image, context_reports=context_reports, structured=structured)
```

---

## Phase 4: Push code to HF Space (~15 min)

### 4.1 Clone the Space repo locally
```bash
git clone https://huggingface.co/spaces/mohamedtaha77/cxr-rag-demo
cd cxr-rag-demo
```

### 4.2 Copy necessary files
Copy from your cxr-rag-system repo:
- `app/app.py` → `app.py` (move to root for HF Spaces)
- `src/` → `src/`
- Updated `app/utils.py` (above) → `app/utils.py`
- New `requirements.txt` (above) → `requirements.txt`

### 4.3 Add README.md for the Space
```markdown
---
title: CXR Intelligence System
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
---

# CXR Intelligence System
Dual-mode chest X-ray analysis: report generation + QA.
```

### 4.4 Push to Space
```bash
git add .
git commit -m "Initial deployment"
git push
```

---

## Phase 5: Wait for build (~15 min)

1. Watch the build logs on your Space page
2. First build downloads all packages + indexes (~5-10 min)
3. Once running, test with a sample image

---

## Troubleshooting

### Build fails with "Out of memory"
→ Add `pip install --no-cache-dir` to requirements

### MedGemma fails to load
→ Check HF_TOKEN secret is set and has gated repo access

### Indexes fail to download
→ Verify INDEX_REPO secret matches your dataset name
→ Make sure HF_TOKEN has read access to the dataset

### ZeroGPU timeout
→ First request always slow (model loading)
→ Subsequent requests fast (~5s)
→ Increase `@spaces.GPU(duration=120)` if needed

---

## Final URL

Your public demo will be at:
```
https://huggingface.co/spaces/mohamedtaha77/cxr-rag-demo
```

Update README.md with this URL after deployment.
