# Running on Kaggle T4 (30 Hours)

## Step 1: Add Kaggle Input (Dataset)

**For EVERY notebook (01, 02, 03, 04):**

1. Open the notebook
2. Click **+ Add Input** in right sidebar (📂 icon)
3. Search: `raddar/chest-xrays-indiana-university`
4. Click **Add**

This mounts the dataset (13 GB) read-only at `/kaggle/input/chest-xrays-indiana-university/` — no download, no disk space used.

## Step 2: Add Kaggle Secrets

In notebook → click **+ Add-ons** → **Secrets**:

| Secret | Where to get |
|--------|-------------|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Step 3: Import Notebooks

Open these one at a time from GitHub:
- `notebooks/01_data_and_qa_dataset.ipynb`
- `notebooks/02_colpali_indexing.ipynb`
- `notebooks/03_pipelines_and_eval.ipynb`
- `notebooks/04_comparison.ipynb`

## Step 4: Run in Order

⚠️ **All outputs save to `/kaggle/working/` (persists on disk after session ends)**

| Notebook | Time | Outputs |
|----------|------|---------|
| **01** | ~30 min | `reports_corpus.csv`, `qa_dataset.jsonl` |
| **02** | ~2 hrs | `colpali_index/`, `clip_index/` |
| **03** | ~3 hrs | `results.csv` |
| **04** | ~1 hr | PNG visualizations |

### Workflow:
1. Run Notebook 01 → outputs save to `/kaggle/working/`
2. **Kill session** (Session menu → Kill session)
3. Run Notebook 02 → loads from disk, saves new outputs
4. **Kill session**
5. Run Notebook 03 → loads everything from disk
6. **Kill session**
7. Run Notebook 04 → loads results, generates visualizations

### Why kill sessions:
- Frees GPU memory between notebooks
- Files on `/kaggle/working/` persist on disk
- Files on `/kaggle/input/` are the dataset (always available when input is added)

## File Locations

```
/kaggle/input/chest-xrays-indiana-university/     # Dataset (read-only)
├── images/                                       # 7,470 PNGs
├── indiana_reports.csv
└── indiana_projections.csv

/kaggle/working/                                  # Your outputs (persists)
├── cxr-rag-system/                              # Cloned repo
├── reports_corpus.csv                           # From Notebook 01
├── qa_dataset.jsonl                             # From Notebook 01
├── colpali_index/                               # From Notebook 02
├── clip_index/                                  # From Notebook 02
├── results.csv                                  # From Notebook 03
└── *.png                                        # Visualizations
```

## Troubleshooting

### "No PNGs in /kaggle/input/chest-xrays-indiana-university"
→ You didn't add the input dataset. Click **+ Add Input** in right sidebar.

### "GROQ_API_KEY not found" / "HF_TOKEN not found"
→ Add secrets via **+ Add-ons** → **Secrets**.

### "No space left on device"
→ You used the old version that downloaded the dataset. Use the latest notebooks from GitHub (they use `/kaggle/input/` directly).

### Image paths broken in Notebook 03/04
→ You forgot to add the input dataset to that session. Add it.

## Time Budget

| Phase | Time |
|-------|------|
| Notebook execution | ~6-7 hours |
| Buffer / reruns | ~23 hours |
| **Total available** | **30 hours** |

You have plenty of time.

## Verification After Each Notebook

```python
import os
WORKING_DIR = '/kaggle/working'

# After Notebook 01
assert os.path.exists(f'{WORKING_DIR}/reports_corpus.csv')
assert os.path.exists(f'{WORKING_DIR}/qa_dataset.jsonl')

# After Notebook 02
assert os.path.exists(f'{WORKING_DIR}/colpali_index/colpali_path_map.pkl')
assert os.path.exists(f'{WORKING_DIR}/clip_index/index.faiss')

# After Notebook 03
assert os.path.exists(f'{WORKING_DIR}/results.csv')

print('✓ All outputs verified')
```
