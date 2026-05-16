"""
CXR Intelligence System — Gradio Dual-Mode Demo (for HuggingFace Spaces ZeroGPU)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import spaces
import torch
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_REPO = os.environ.get("INDEX_REPO", "mohamedtaha77/cxr-rag-indexes")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
INDEX_DIR = "/tmp/cxr_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)


# ── Download indexes from HF Dataset (one-time) ───────────────────────────────
def download_indexes():
    files = [
        "colpali_index/colpali_embeddings.pt",
        "colpali_index/colpali_paths.pkl",
        "clip_index/clip_faiss.index",
        "clip_index/clip_paths.pkl",
        "reports_corpus.csv",
    ]
    for f in files:
        target = os.path.join(INDEX_DIR, f)
        if not os.path.exists(target):
            hf_hub_download(
                repo_id=INDEX_REPO,
                filename=f,
                repo_type="dataset",
                local_dir=INDEX_DIR,
                token=HF_TOKEN,
            )
    print("✓ Indexes downloaded")


download_indexes()

# Load corpus once
CORPUS_PATH = os.path.join(INDEX_DIR, "reports_corpus.csv")
corpus_df = pd.read_csv(CORPUS_PATH)
path_to_impression = dict(zip(corpus_df["image_path"], corpus_df["impression"]))


# ── Model loaders (lazy, called inside GPU function) ──────────────────────────
_colpali = None
_clip = None
_generator = None


def get_colpali():
    global _colpali
    if _colpali is None:
        from src.retrieval.colpali_retriever import ColPaliRetriever
        _colpali = ColPaliRetriever.from_index(os.path.join(INDEX_DIR, "colpali_index"))
    return _colpali


def get_clip():
    global _clip
    if _clip is None:
        from src.retrieval.clip_retriever import CLIPRetriever
        _clip = CLIPRetriever()
        _clip.load_index(os.path.join(INDEX_DIR, "clip_index"))
    return _clip


def get_generator():
    global _generator
    if _generator is None:
        from src.generation.medgemma_generator import MedGemmaGenerator
        _generator = MedGemmaGenerator(hf_token=HF_TOKEN, load_in_4bit=True)
    return _generator


# ── Inference functions (decorated for ZeroGPU) ───────────────────────────────
@spaces.GPU(duration=120)
def generate_report(image, retriever_choice, use_rag, top_k):
    if image is None:
        return "Please upload a CXR image.", None, None, None

    image = image.convert("RGB")
    generator = get_generator()

    context_reports = []
    retrieved_images = []
    retrieved_scores = []

    if use_rag:
        retriever = get_colpali() if "ColPali" in retriever_choice else get_clip()
        if "ColPali" in retriever_choice:
            results = retriever.search("chest x-ray radiology findings", k=top_k)
        else:
            results = retriever.search_by_text("chest x-ray radiology findings", k=top_k)

        for r in results:
            impression = path_to_impression.get(r.get("image_path", ""), "")
            if impression:
                context_reports.append(impression)
            if r.get("image"):
                retrieved_images.append(r["image"])
                retrieved_scores.append(f"Score: {r['score']:.3f}")

    report = generator.generate_report(
        image, context_reports=context_reports if use_rag else None
    )

    # Pad to 3 images for display
    while len(retrieved_images) < 3:
        retrieved_images.append(None)
        retrieved_scores.append("")

    return (
        report,
        retrieved_images[0],
        retrieved_images[1],
        retrieved_images[2],
    )


@spaces.GPU(duration=120)
def answer_question(image, question, retriever_choice, top_k):
    if image is None:
        return "Please upload a CXR image.", None, None, None
    if not question.strip():
        return "Please enter a clinical question.", None, None, None

    image = image.convert("RGB")
    generator = get_generator()
    retriever = get_colpali() if "ColPali" in retriever_choice else get_clip()

    if "ColPali" in retriever_choice:
        results = retriever.search(question, k=top_k)
    else:
        results = retriever.search_by_text(question, k=top_k)

    context_reports = []
    retrieved_images = []
    for r in results:
        impression = path_to_impression.get(r.get("image_path", ""), "")
        if impression:
            context_reports.append(impression)
        if r.get("image"):
            retrieved_images.append(r["image"])

    if not context_reports:
        context_reports = ["No relevant context available."]

    answer = generator.answer_question(image, question, context_reports)

    while len(retrieved_images) < 3:
        retrieved_images.append(None)

    return (
        answer,
        retrieved_images[0],
        retrieved_images[1],
        retrieved_images[2],
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="CXR Intelligence System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🫁 Multi-Modal Chest X-Ray Intelligence System
        **DSAI 413 — Assignment 2** | Mohammed Taha

        Upload a chest X-ray to generate a structured radiology report,
        or ask a clinical question about the findings.
        """
    )

    with gr.Tabs():
        # ── Report Generation Tab ─────────────────────────────────────────────
        with gr.Tab("📝 Report Generation"):
            with gr.Row():
                with gr.Column():
                    img_input_a = gr.Image(type="pil", label="Upload CXR Image")
                    retriever_a = gr.Dropdown(
                        ["ColPali v1.3 (Primary)", "CLIP ViT-L/14 (Baseline)"],
                        value="ColPali v1.3 (Primary)",
                        label="Retrieval Model",
                    )
                    use_rag_a = gr.Checkbox(value=True, label="Enable RAG")
                    top_k_a = gr.Slider(1, 5, value=3, step=1, label="Retrieved cases (k)")
                    gen_btn = gr.Button("Generate Report", variant="primary")
                with gr.Column():
                    report_output = gr.Textbox(label="Generated Report", lines=10)
                    with gr.Row():
                        ret_img_1 = gr.Image(label="Retrieved #1", interactive=False)
                        ret_img_2 = gr.Image(label="Retrieved #2", interactive=False)
                        ret_img_3 = gr.Image(label="Retrieved #3", interactive=False)
            gen_btn.click(
                generate_report,
                inputs=[img_input_a, retriever_a, use_rag_a, top_k_a],
                outputs=[report_output, ret_img_1, ret_img_2, ret_img_3],
            )

        # ── QA Mode Tab ───────────────────────────────────────────────────────
        with gr.Tab("❓ QA Mode"):
            with gr.Row():
                with gr.Column():
                    img_input_b = gr.Image(type="pil", label="Upload CXR Image")
                    question_input = gr.Textbox(
                        label="Clinical Question",
                        placeholder="Is there evidence of pleural effusion?",
                        lines=2,
                    )
                    retriever_b = gr.Dropdown(
                        ["ColPali v1.3 (Primary)", "CLIP ViT-L/14 (Baseline)"],
                        value="ColPali v1.3 (Primary)",
                        label="Retrieval Model",
                    )
                    top_k_b = gr.Slider(1, 5, value=3, step=1, label="Retrieved cases (k)")
                    qa_btn = gr.Button("Get Answer", variant="primary")

                    gr.Examples(
                        examples=[
                            "Is there evidence of pleural effusion?",
                            "Does this chest X-ray show signs of cardiomegaly?",
                            "Are there any support devices visible?",
                            "Is there evidence of atelectasis?",
                            "Are the lung fields clear?",
                            "Is there a pneumothorax present?",
                        ],
                        inputs=question_input,
                    )
                with gr.Column():
                    answer_output = gr.Textbox(label="Answer", lines=5)
                    with gr.Row():
                        ret_img_4 = gr.Image(label="Evidence #1", interactive=False)
                        ret_img_5 = gr.Image(label="Evidence #2", interactive=False)
                        ret_img_6 = gr.Image(label="Evidence #3", interactive=False)
            qa_btn.click(
                answer_question,
                inputs=[img_input_b, question_input, retriever_b, top_k_b],
                outputs=[answer_output, ret_img_4, ret_img_5, ret_img_6],
            )

    gr.Markdown(
        """
        ---
        **About**: ColPali uses patch-level late-interaction retrieval. CLIP uses global embeddings.
        MedGemma 1.5 4B IT (4-bit) generates reports/answers. First request takes ~1 min (model load).
        """
    )


if __name__ == "__main__":
    demo.launch()
