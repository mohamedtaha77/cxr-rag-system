# Project Errors Log

Comprehensive log of errors faced during the CXR RAG System project, organized by phase.

---

## Phase 1: Data Loading (Notebook 01)

### 1. OpenI Direct Download URL Broken
**Error:** `https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.zip` returned 404.
**Cause:** NIH/NLM deprecated the direct download URL.
**Fix:** Switched to Kaggle dataset `raddar/chest-xrays-indiana-university` which provides the same data with CSV reports.

### 2. OpenILoader XML Parsing Empty DataFrame
**Error:** `KeyError: 'impression'` and `KeyError: 'image_path'` when loading data.
**Cause:** XML image IDs (e.g., `IM-0001-0001.dcm`) didn't match PNG filenames on disk.
**Fix:** Rewrote `OpenILoader` to add `load_from_kaggle_csvs()` method that reads `indiana_reports.csv` + `indiana_projections.csv` directly.

### 3. Git Commit Author Mismatch
**Error:** Commits showed generic name instead of GitHub profile.
**Cause:** Git config email didn't match GitHub account email.
**Fix:** Set git config to correct email and used `--reset-author` flag.

### 4. Module Import Caching in Notebooks
**Error:** Old class signature still in memory after pulling new code.
**Cause:** Python caches imported modules per session.
**Fix:** Added `importlib.reload()` + `del sys.modules['module_name']` before re-importing.

### 5. Undefined KAGGLE_DIR Variable
**Error:** `NameError: name 'KAGGLE_DIR' is not defined`.
**Cause:** Variable defined in a cell that wasn't re-run.
**Fix:** Added `KAGGLE_DIR = '/content/openi'` explicitly at top of cell.

### 6. Groq Rate Limit (429 Too Many Requests)
**Error:** `HTTPStatusError: Client error '429 Too Many Requests'` during QA generation.
**Cause:** Free tier limit is 28 requests/minute. Original code had only 1-4s exponential backoff which wasn't enough.
**Fix:** Improved `_rate_limit()` in `qa_creator.py` to spread requests smoothly across 60s window. Increased backoff to 10/20/40 seconds.

### 7. Colab Session Timeout
**Error:** Colab free tier session expiring during long QA generation (~3 hours needed for full dataset).
**Cause:** Colab free tier has ~12 hour limit per session.
**Fix:** Reduced `max_studies` from 500 to 200, then switched to Kaggle (30 hours/week).

---

## Phase 2: Indexing (Notebook 02)

### 8. faiss-gpu Package Not Found
**Error:** `ERROR: Could not find a version that satisfies the requirement faiss-gpu`.
**Cause:** `faiss-gpu` was removed from PyPI.
**Fix:** Switched to `faiss-cpu` (negligible performance difference for our use case).

### 9. ModuleNotFoundError: No module named 'src'
**Error:** Cannot import `src.retrieval.colpali_retriever` despite `sys.path.insert()`.
**Cause:** `sys.path` modifications weren't persisting between cells in some cases.
**Fix:** Multiple approaches:
- Combined setup + imports into single cell
- Used `importlib.util.spec_from_file_location()` for direct loading
- Added `sys.path.insert(0, REPO_PATH)` before any `from src...` imports

### 10. torchao Version Incompatibility
**Error:** `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`.
**Cause:** Colab/Kaggle preinstalled older torchao than ColPali required.
**Fix:** `!pip install -q --upgrade torchao`.

### 11. ColPali Initial OOM (Caching Allocator Warmup)
**Error:** `OutOfMemoryError: CUDA out of memory. Tried to allocate 5.36 GiB.`
**Cause:** Transformers' caching allocator warmup tried to pre-allocate large memory block.
**Fix:** Cleared GPU cache before loading + restarted Colab runtime.

### 12. byaldi: Unsupported File Type
**Error:** `ValueError: Unsupported input type:` during byaldi indexing.
**Cause:** byaldi's `model.index()` recursively scanned directory and tried to process non-image files (or files with empty extension).
**Fix:** Filtered file list to only PNG/JPG before indexing.

### 13. byaldi: 'RAGMultiModalModel' object has no attribute 'add_file'
**Error:** Tried to use `add_file()` method to add images one-by-one.
**Cause:** This method doesn't exist in byaldi API.
**Fix:** Reverted to `model.index(input_path=...)` but later replaced byaldi entirely (see #15).

### 14. byaldi: Extremely Slow Indexing (44s/image)
**Error:** ColPali indexing took 8 hours for only 650/7470 images.
**Cause:** byaldi saved entire index to disk after every single image (O(n²) disk writes).
**Fix:** Completely replaced byaldi with direct `colpali-engine` usage:
- Batch encoding on GPU
- Single save at end of indexing
- Speed improved from 44s/image to ~2.6s/image (~17x faster)

### 15. peft Version Incompatibility
**Error:** `ImportError: cannot import name '_maybe_shard_state_dict_for_tp' from 'peft.utils.save_and_load'`.
**Cause:** `transformers 4.50+` requires `_maybe_shard_state_dict_for_tp` from `peft 0.15+`. Older peft missing this function.
**Fix:** `!pip install -q --upgrade peft transformers`.

### 16. CUDA Out of Memory During ColPali Indexing
**Error:** `OutOfMemoryError: Tried to allocate 3.95 GiB. GPU 0 has 14.56 GiB total, 3.39 GiB free`.
**Cause:** Batch size of 4 too large for T4. Also, model was computing loss (`labels` in inputs).
**Fix:**
- Reduced `BATCH_SIZE` from 4 to 1
- Added `batch_inputs.pop('labels', None)` to skip loss computation
- Periodic `torch.cuda.empty_cache()` every 100 images

---

## Phase 3: Kaggle Migration

### 17. Kaggle Working Dir Full
**Error:** `OSError: [Errno 28] No space left on device` when downloading dataset.
**Cause:** Dataset is 13.2 GB; `/kaggle/working/` has limited space.
**Fix:** Use Kaggle Input feature (`raddar/chest-xrays-indiana-university`) which mounts the dataset read-only at `/kaggle/input/` — no disk space used.

### 18. Kaggle Input Path Auto-Detection
**Error:** Dataset mounted at unexpected path `/kaggle/input/datasets/raddar/chest-xrays-indiana-university/` instead of `/kaggle/input/chest-xrays-indiana-university/`.
**Cause:** Kaggle mount path varies depending on dataset addition method.
**Fix:** Used `glob.glob('/kaggle/input/**/indiana_reports.csv', recursive=True)` to auto-detect the actual location.

### 19. Kaggle Input is Read-Only (git pull failed)
**Error:** `git pull` failed with exit status 128 when run inside `/kaggle/input/...`.
**Cause:** Kaggle Input filesystem is read-only.
**Fix:** Always clone the repo to `/kaggle/working/cxr-rag-system/` (writable) instead of using the repo snapshot from input.

### 20. medgemma_generator.py Relative Import Failure
**Error:** `ModuleNotFoundError: No module named 'src'` when loading `medgemma_generator.py`.
**Cause:** File contains `from src.generation.prompts import ...` but src wasn't in sys.path when loaded via importlib.
**Fix:** Use standard `sys.path.insert(0, REPO_PATH)` + normal Python imports instead of `importlib.util.spec_from_file_location()`.

---

## Phase 4: Evaluation (Notebook 03)

### 21. HuggingFace 403 Forbidden (MedGemma)
**Error:** `HfHubHTTPError: 403 Forbidden... Please enable access to public gated repositories in your fine-grained token settings`.
**Cause:** Two issues:
1. User hadn't accepted MedGemma's gated model terms on HuggingFace.
2. HF_TOKEN didn't have "Read access to public gated repos" permission.
**Fix:**
1. Visit `https://huggingface.co/google/medgemma-1.5-4b-it` → click "Agree and access repository".
2. Create new token with **"Read access to contents of all public gated repos you can access"** permission checked.
3. Update Kaggle secret with new token.

### 22. PyTorch CUDA Not Available
**Error:** `AssertionError: Torch not compiled with CUDA enabled`.
**Cause:** GPU accelerator not enabled in Kaggle notebook settings, OR pip install replaced GPU torch with CPU torch.
**Fix:**
1. Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x2.
2. If still failing: `!pip install -q torch --index-url https://download.pytorch.org/whl/cu121`.
3. Restart kernel.

### 23. BERTScore OverflowError
**Error:** `OverflowError: int too big to convert` in `tokenizer.enable_truncation()`.
**Cause:** Newer transformers store `model_max_length` as `sys.maxsize` which overflows Rust tokenizer's int conversion.
**Fix:** Replaced BERTScore with sentence-transformer cosine similarity (MiniLM) — equivalent semantic comparison without the tokenizer issue.

### 24. Identical Predictions Across All Systems
**Error:** All 3 systems (ColPali, CLIP, Direct) produced identical reports → identical metrics.
**Cause:** `study_id` extraction from filename was wrong. Filename like `349_IM-1697-2001.dcm.png` was parsed to `349_IM-1697-2001` but corpus `study_id` was just `349`. Lookup returned empty string → context was empty → all systems used direct prompt.
**Fix:** Changed lookup to use `image_path` directly:
```python
path_to_impression = dict(zip(corpus_df['image_path'], corpus_df['impression']))
context = [path_to_impression.get(r['image_path'], '') for r in retrieved]
```

### 25. RadGraph Module Not Found (Expected)
**Error:** `[RadGraph skipped] No module named 'radgraph'`.
**Cause:** RadGraph requires PhysioNet credentials and complex setup.
**Fix:** Intentionally skipped — RadGraph was optional. Used BERTScore + ROUGE-L as primary metrics.

---

## Phase 5: Session & Workflow

### 26. Kaggle Session File Sharing
**Issue:** Outputs from one notebook session aren't automatically available in another notebook.
**Cause:** Each Kaggle notebook has its own `/kaggle/working/` directory.
**Fix:** Use **Quick Save** to save notebook version with outputs, then add as "Notebook Output" input to next notebook.

### 27. Combined Notebook Workflow
**Issue:** Running Notebook 02 in same session as Notebook 01 risked overwriting outputs.
**Fix:** Quick Save creates new versions, doesn't overwrite. Naming convention used: `01+02 - Data + QA + Indexes Complete`.

---

## Phase 6: QA Evaluation

### 28. BLEU-4 ZeroDivisionError
**Error:** `ZeroDivisionError: Fraction(0, 0)` in `evaluate_qa` when computing BLEU-4.
**Cause:** BLEU-4 fails when predictions are shorter than 4 tokens (no 4-grams to score) or when no n-gram overlap exists between predictions and references.
**Fix:** Bypassed `evaluate_qa()` and computed BERTScore + ROUGE-L directly. Filtered out empty predictions before scoring:
```python
valid = [(p, r) for p, r in zip(qa_preds, qa_refs) if p and r]
```

### 29. QA Dataset Had No Test Split
**Issue:** All 1,515 generated QA pairs had `split == 'train'`, none in `val` or `test`.
**Cause:** QA generation was limited to first 200 studies, which all fell within the 80% train split of the 3,955-study corpus (sorted before splitting).
**Fix:** Used first 30 train-split QA pairs for evaluation, documented limitation in REPORT.md.

---

## Phase 7: Deployment (HF Spaces / Kaggle ngrok)

### 30. HuggingFace Token Lacks Write Permission
**Error:** `403 Forbidden: You have read access but not the required permissions for this operation` when uploading to HF Dataset.
**Cause:** Original HF_TOKEN was created with only read permissions (for MedGemma gated access).
**Fix:** Created new fine-grained token with:
- ✓ Read access to contents of all repos under your personal namespace
- ✓ Read access to contents of all public gated repos you can access
- ✓ Write access to contents/settings of all repos under your personal namespace

### 31. ZeroGPU Requires HF PRO Subscription
**Error:** "Subscribe to PRO at https://huggingface.co/subscribe/pro to use ZeroGPU" when creating a Space.
**Cause:** HuggingFace changed ZeroGPU access policy — now requires PRO ($9/month).
**Fix:** Switched deployment strategy to **Kaggle T4 + ngrok** for free GPU access. Public URL available while Kaggle session alive.

### 32. ZeroGPU Streamlit Incompatibility
**Issue:** "ZeroGPU is only available with Gradio SDK" warning when selecting Streamlit.
**Cause:** ZeroGPU's `@spaces.GPU` decorator only works with Gradio function wrappers.
**Fix:** Created `app/app_gradio.py` as a Gradio alternative to the Streamlit app. Kept Streamlit version for local deployment.

### 33. huggingface_hub KernelInfo Import Error
**Error:** `ImportError: cannot import name 'KernelInfo' from 'huggingface_hub.hf_api'`.
**Cause:** Partial/inconsistent install of huggingface_hub on Kaggle — `_snapshot_download.py` tried to import a class missing from `hf_api.py`.
**Fix:** Forced clean reinstall:
```bash
pip install -q --upgrade --force-reinstall huggingface_hub
```
Then restart kernel to load the consistent version.

### 34. ngrok Connection Refused (Gradio Died Silently)
**Error:** `ERR_NGROK_8012: dial tcp [::1]:7860: connect: connection refused`. ngrok tunnel works but no process on port 7860.
**Cause:** `subprocess.Popen` to launch Gradio captured stdout/stderr silently — when Gradio crashed (due to error #33), the failure was invisible.
**Fix:** Run Gradio in a **background thread** in the same Python process instead of subprocess. Errors appear in cell output:
```python
import threading
def run_gradio():
    app_mod.demo.launch(server_port=7860, server_name='0.0.0.0', quiet=True)
threading.Thread(target=run_gradio, daemon=True).start()
```

### 35. Evaluation CSVs Ignored by Git
**Error:** `git status` showed no changes despite `evaluation/results.csv` etc. existing locally.
**Cause:** `.gitignore` had `evaluation/*.csv` blocking commits.
**Fix:** Removed that line from `.gitignore` so final evaluation results could be tracked in version control.

---

## Common Patterns

### Module Reloading After Pull
When pulling new code from GitHub mid-session, modules must be explicitly reloaded:
```python
import importlib.util
spec = importlib.util.spec_from_file_location("name", "/path/to/file.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ClassName = mod.ClassName
```

### Memory Management Between Models
Always free VRAM between loading different models:
```python
del retriever
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```

### Path-Based Lookups (Not ID-Based)
When dealing with retrieved images, use `image_path` as the lookup key rather than parsing study IDs from filenames — filename formats vary.

---

## Key Lessons Learned

1. **Don't trust direct URLs**: NIH/NLM URLs change; prefer Kaggle/HuggingFace mirrors.
2. **Test before scaling**: byaldi seemed fine for small tests but was unusable at scale.
3. **Always batch on GPU**: Per-image processing is 10-50x slower than batched.
4. **Verify retrieval is being used**: Empty context = silent fallback to direct generation.
5. **Auto-detect paths**: Kaggle Input mount paths vary; use `glob` patterns.
6. **Pin compatible versions**: peft + transformers + colpali-engine version conflicts are common.
7. **Use Kaggle Inputs**: For 13+ GB datasets, mount instead of download.
8. **Quick Save preserves state**: Don't kill sessions before saving — outputs only persist with explicit save versions.
