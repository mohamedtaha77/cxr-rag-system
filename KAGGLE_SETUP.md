# Running on Kaggle T4 (30 Hours)

All 4 notebooks are now optimized for Kaggle's T4 GPU. Here's the complete setup:

## Step 1: Add Kaggle Secrets

In your Kaggle notebook, click **Secrets** (🔑 icon on left sidebar) and add:

| Secret | Value | Where to get |
|--------|-------|-------------|
| `GROQ_API_KEY` | Your Groq API key | [console.groq.com](https://console.groq.com) |
| `KAGGLE_USERNAME` | Your Kaggle username | [kaggle.com/settings/account](https://www.kaggle.com/settings/account) |
| `KAGGLE_KEY` | Your Kaggle API key | [kaggle.com/settings/api](https://www.kaggle.com/settings/api) → Create New Token |
| `HF_TOKEN` | HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

## Step 2: Import Notebooks

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create new notebook
3. Click **File** → **Import notebook**
4. Enter: `https://github.com/mohamedtaha77/cxr-rag-system`
5. Select notebook 01 → Import

Repeat for notebooks 02, 03, 04.

**OR** download `.ipynb` files directly from GitHub `/notebooks` folder.

## Step 3: Run in Order

| Notebook | Time | What it does |
|----------|------|------------|
| **01** | ~2.5 hrs | Downloads dataset, generates QA pairs via Groq |
| **02** | ~2 hrs | Builds ColPali + CLIP indexes |
| **03** | ~3 hrs | Runs all 3 systems, computes evaluation metrics |
| **04** | ~1 hr | Visualization + comparison |

**Total execution: ~8-9 hours** → 20+ hours buffer for reruns.

## Key Changes from Colab

### Paths
- **Old**: `/content/drive/MyDrive/cxr_rag` → **New**: `/kaggle/working`
- All data persists in `/kaggle/working/` automatically

### Secrets
- **Old**: `userdata.get('KEY')` → **New**: `os.environ.get('KEY')`
- Kaggle auto-populates secrets into environment

### Packages
- Changed `faiss-gpu` → `faiss-cpu` (faiss-gpu doesn't exist on PyPI)

## Data Locations (All in `/kaggle/working/`)

```
/kaggle/working/
├── openi/                    # Downloaded dataset
│   ├── images/              # 7,470 PNG files
│   ├── indiana_reports.csv
│   └── indiana_projections.csv
├── reports_corpus.csv        # Parsed reports (from Notebook 01)
├── qa_dataset.jsonl          # Generated QA pairs (from Notebook 01)
├── colpali_index/            # Built in Notebook 02
├── clip_index/               # Built in Notebook 02
├── results.csv               # Metrics (from Notebook 03)
├── cxr-rag-system/          # Cloned repo
└── *.png files              # Visualizations from Notebooks 02, 04
```

## Troubleshooting

### "No PNG images found"
→ Notebook 01 didn't finish. Re-run it completely.

### "GROQ_API_KEY not found"
→ Check you added the secret to notebook secrets (🔑), not user account settings.

### Out of memory errors
→ Kaggle T4 has 15 GB VRAM (same as Colab). Should not happen. Check VRAM with:
```python
import torch
print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB')
```

### Build times are slower than expected
→ Normal. Kaggle T4 speed varies by load. You have 30 hours, so plenty of buffer.

## Verification Checklist

After each notebook:
- ✓ Notebook 01: Check `/kaggle/working/reports_corpus.csv` exists (~3 MB)
- ✓ Notebook 02: Check `/kaggle/working/colpali_index/` has files
- ✓ Notebook 02: Check `/kaggle/working/clip_index/` has files
- ✓ Notebook 03: Check `/kaggle/working/results.csv` exists (~1 KB)
- ✓ Notebook 04: Check visualization PNG files exist

## Done?

When all 4 notebooks complete:
1. Download `results.csv` from your notebook output
2. Update README.md with actual metrics
3. Commit and push to GitHub

---

**Questions?** Check the notebook output for detailed error messages. All notebooks print progress regularly.
