# Live Demo on Kaggle + ngrok

Get a **public URL** for your Gradio app while running on Kaggle T4 GPU.

Total setup time: **~15 minutes**.

---

## Step 1: Get free ngrok account (~3 min)

1. Go to https://dashboard.ngrok.com/signup
2. Sign up with email (free tier is fine)
3. After signup, go to https://dashboard.ngrok.com/get-started/your-authtoken
4. **Copy your authtoken** (looks like `2abc...xyz`)

## Step 2: Add ngrok token to Kaggle secrets

In a new (or existing) Kaggle notebook:
1. **Add-ons** → **Secrets**
2. Click **Add a new secret**
3. Name: `NGROK_TOKEN`
4. Value: your authtoken
5. Check "Attach to notebook"
6. Save

## Step 3: Add required inputs to your Kaggle notebook

In the right sidebar → **+ Add Input**:
1. ✓ Kaggle Dataset: `raddar/chest-xrays-indiana-university`
2. ✓ Notebook Output: `01+02 - Data + QA + Indexes Complete` (your saved notebook with indexes)

## Step 4: Run the demo cell

Create one cell with this exact code:

```python
import os, sys, subprocess, time, glob

# ── Install packages ──────────────────────────────────────────────────────────
subprocess.run(['pip', 'install', '-q', '--upgrade', 'peft', 'transformers'], check=True)
subprocess.run(['pip', 'install', '-q', 'gradio', 'pyngrok', 'colpali-engine',
                'accelerate', 'bitsandbytes', 'open-clip-torch', 'faiss-cpu'], check=True)
subprocess.run(['pip', 'install', '-q', '--upgrade', 'torchao'], check=True)

# ── Clone repo ────────────────────────────────────────────────────────────────
REPO = '/kaggle/working/cxr-rag-system'
if not os.path.exists(REPO):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/mohamedtaha77/cxr-rag-system.git', REPO], check=True)
else:
    subprocess.run(['git', '-C', REPO, 'pull', '-q'], check=True)

# ── Set env vars for app ──────────────────────────────────────────────────────
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

os.environ['HF_TOKEN'] = secrets.get_secret('HF_TOKEN')

# Point app at the local index files (instead of downloading from HF Dataset)
prev_output = next(iter(glob.glob('/kaggle/input/notebooks/**/colpali_index', recursive=True)), None)
if not prev_output:
    raise RuntimeError('Add the 01+02 notebook output as Input first')

INDEX_BASE = os.path.dirname(prev_output)
print(f'Index base: {INDEX_BASE}')

# Override the download function to skip downloading (files already on Kaggle)
os.environ['SKIP_DOWNLOAD'] = '1'
os.environ['LOCAL_INDEX_BASE'] = INDEX_BASE

# ── Patch app to use local indexes ────────────────────────────────────────────
app_path = f'{REPO}/app/app_gradio.py'
with open(app_path, 'r') as f:
    content = f.read()

# Replace INDEX_DIR with local Kaggle path
content = content.replace(
    'INDEX_DIR = "/tmp/cxr_indexes"',
    f'INDEX_DIR = "{INDEX_BASE}"'
)
content = content.replace(
    'download_indexes()',
    '# download_indexes()  # skipped, using local files'
)

# Save patched version
patched_path = '/kaggle/working/app_kaggle.py'
with open(patched_path, 'w') as f:
    f.write(content)

# ── Launch Gradio in background ───────────────────────────────────────────────
sys.path.insert(0, REPO)
gradio_proc = subprocess.Popen(['python', patched_path],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for Gradio to start
print('Starting Gradio... (~30 sec)')
time.sleep(30)

# ── Setup ngrok tunnel ────────────────────────────────────────────────────────
from pyngrok import ngrok
ngrok.set_auth_token(secrets.get_secret('NGROK_TOKEN'))

# Kill any existing tunnels
for tunnel in ngrok.get_tunnels():
    ngrok.disconnect(tunnel.public_url)

public_url = ngrok.connect(7860)
print(f'\n{"="*60}')
print(f'✓ PUBLIC URL: {public_url}')
print(f'{"="*60}')
print('\nThis URL will work as long as this Kaggle session is alive.')
print('First request will be slow (~60s — models loading).')
```

## Step 5: Open the public URL

The cell prints something like:
```
✓ PUBLIC URL: NgrokTunnel: "https://abc-123-456.ngrok-free.app" -> "http://localhost:7860"
```

Open that URL in a new browser tab. You'll see your Gradio app live.

## Step 6: Record your demo

1. Open the public URL
2. Start screen recording (Loom, OBS, or built-in tools)
3. Demo:
   - Upload a sample CXR
   - Click "Generate Report" (wait ~60s first time, then ~10s)
   - Show the retrieved evidence images
   - Switch to QA tab
   - Ask "Is there evidence of pleural effusion?"
   - Show the answer + evidence

## Important notes

- **Free ngrok tier**: Tunnel URL changes each session. Add a custom domain in ngrok dashboard if you want a stable URL.
- **Kaggle session**: ~9 hours max. URL dies when session ends.
- **First request slow**: Model loading takes ~60s. Subsequent requests ~10s.
- **For your demo video**: Record the second request, not the first (faster).

## Troubleshooting

### "RuntimeError: Add the 01+02 notebook output as Input first"
→ Click **+ Add Input** in Kaggle sidebar, find your `01+02 - Data + QA + Indexes Complete` notebook

### Gradio doesn't start
→ Check the gradio_proc output:
```python
print(gradio_proc.stdout.read().decode())
print(gradio_proc.stderr.read().decode())
```

### ngrok says "tunnel limit exceeded"
→ Free tier allows 1 tunnel at a time. Kill old tunnels first:
```python
from pyngrok import ngrok
ngrok.kill()
```

### Out of memory
→ Restart Kaggle kernel and try again
→ ColPali + MedGemma should fit in 15GB T4 VRAM

---

## Demo URL example

After running, you'll have something like:
```
https://9b3c-34-148-92-17.ngrok-free.app
```

Add this to your README and demo video.
