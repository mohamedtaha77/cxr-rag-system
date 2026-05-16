# HuggingFace Spaces Deployment — Gradio + ZeroGPU

**ZeroGPU is Gradio-only** (Streamlit doesn't get free GPU on Spaces).
We use Gradio + ZeroGPU for free A10G GPU access.

Estimated time: **~1 hour** if you follow exactly.

---

## Phase 1: Upload indexes to HF Dataset ✓ DONE
Already uploaded to `mohamedtaha77/cxr-rag-indexes`.

---

## Phase 2: Create Gradio Space

### 2.1 Create Space
1. Go to https://huggingface.co/new-space
2. Owner: `mohamedtaha77`
3. Space name: `cxr-rag-demo`
4. License: `mit`
5. SDK: **Gradio** (not Streamlit!)
6. Hardware: **ZeroGPU - Nvidia A10G**
7. Visibility: **Public**
8. Click **Create Space**

### 2.2 Add Secrets
Settings → Variables and secrets → Add:
- `HF_TOKEN` = your write token
- `INDEX_REPO` = `mohamedtaha77/cxr-rag-indexes`

---

## Phase 3: Prepare files

### 3.1 Clone the Space repo locally
```bash
git clone https://huggingface.co/spaces/mohamedtaha77/cxr-rag-demo
cd cxr-rag-demo
```

### 3.2 Copy these files from your cxr-rag-system repo:

**1. Main app file** (rename when copying):
- `cxr-rag-system/app/app_gradio.py` → `app.py` (in Space root)

**2. Source code:**
- `cxr-rag-system/src/` → `src/`

**3. Create `requirements.txt`** in Space root with:
```
gradio
spaces
torch
transformers>=4.45
peft>=0.14
accelerate
bitsandbytes
colpali-engine
open-clip-torch
faiss-cpu
huggingface_hub
pandas
pillow
```

**4. Create README.md** in Space root:
```markdown
---
title: CXR Intelligence System
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
---

# CXR Intelligence System
Dual-mode chest X-ray analysis using ColPali + MedGemma.
- Report Generation: CXR → structured radiology report
- QA Mode: CXR + question → evidence-grounded answer
```

### 3.3 Final structure
```
cxr-rag-demo/
├── app.py              ← copied from app/app_gradio.py
├── requirements.txt
├── README.md
└── src/
    ├── retrieval/
    │   ├── colpali_retriever.py
    │   └── clip_retriever.py
    ├── generation/
    │   ├── medgemma_generator.py
    │   └── prompts.py
    ├── evaluation/
    ├── data/
    └── __init__.py    (empty file)
```

Make sure every folder under `src/` has an `__init__.py` (even empty).

---

## Phase 4: Push to HF Space

```bash
git add .
git commit -m "Initial deployment"
git push
```

If `git push` asks for credentials, use your **HF username** + **HF write token** (not password).

---

## Phase 5: Wait for build (~10-15 min)

1. Watch build logs on your Space page
2. First build: installs all packages (~5 min)
3. App starts: downloads indexes from Dataset (~2 min)
4. First request: loads ColPali + MedGemma (~60 sec)
5. Subsequent requests: ~5-10 sec

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
→ Make sure `src/__init__.py` and `src/*/__init__.py` exist (empty files OK)

### "Cannot access google/medgemma..."
→ HF_TOKEN secret missing or lacks gated repo permission
→ Recreate token with "Read access to public gated repos" + "Write access to your repos"

### "OutOfMemoryError"
→ ZeroGPU has 24 GB VRAM, should fit ColPali (6GB) + MedGemma 4-bit (3GB)
→ Make sure `@spaces.GPU(duration=120)` decorator is on inference functions

### App crashes on startup
→ Check build logs for missing dependencies
→ Verify all files in `src/` are committed

### Indexes fail to download
→ Verify INDEX_REPO secret matches your dataset name exactly
→ HF_TOKEN must have read access to the dataset

---

## Final URL

After successful deployment:
```
https://huggingface.co/spaces/mohamedtaha77/cxr-rag-demo
```

Update README.md and demo video with this link.
