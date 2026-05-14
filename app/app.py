"""
CXR Intelligence System — Streamlit Dual-Mode Demo
Modes:
  1. Report Generation — upload a CXR → get structured radiology report
  2. QA Mode          — upload a CXR + ask a clinical question → evidence-grounded answer
"""
import os
import sys
import io

import streamlit as st
from PIL import Image

# Allow importing src/ from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import get_retriever, load_medgemma, load_corpus, study_id_from_path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR Intelligence System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("CXR Intelligence System")
st.sidebar.caption("DSAI 413 — Assignment 2")

mode = st.sidebar.radio(
    "Select Mode",
    ["Report Generation", "QA Mode"],
    index=0,
)
retriever_choice = st.sidebar.selectbox(
    "Retrieval Model",
    ["ColPali v1.3 (Primary)", "CLIP ViT-L/14 (Baseline)"],
)
use_rag = st.sidebar.toggle("Enable RAG", value=True, help="Augment generation with retrieved similar cases")
top_k   = st.sidebar.slider("Retrieved cases (k)", 1, 5, 3)

st.sidebar.markdown("---")
st.sidebar.markdown("**About this system**")
st.sidebar.markdown(
    "- **ColPali**: late-interaction retrieval over image patches\n"
    "- **MedGemma**: Google's medical vision-language model\n"
    "- **CLIP**: global embedding baseline for comparison"
)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Multi-Modal Chest X-Ray Intelligence System")
st.markdown(
    "Upload a chest X-ray to generate a structured radiology report "
    "or ask a clinical question about the findings."
)

uploaded_file = st.file_uploader(
    "Upload Chest X-Ray (JPG or PNG)",
    type=["jpg", "jpeg", "png"],
    help="Frontal (PA or AP) chest X-ray works best",
)

if uploaded_file is None:
    st.info("Upload a chest X-ray image to get started.")
    st.stop()

# Load the image
raw_bytes = uploaded_file.read()
image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

# ── Load models (cached) ──────────────────────────────────────────────────────
retriever          = get_retriever(retriever_choice)
generator          = load_medgemma()
study_to_impression = load_corpus()

# ── Mode: Report Generation ───────────────────────────────────────────────────
if mode == "Report Generation":
    st.header("Report Generation Mode")
    structured = st.checkbox("Structured JSON output", value=False)

    col_img, col_report = st.columns([1, 1], gap="large")
    with col_img:
        st.subheader("Input CXR")
        st.image(image, use_column_width=True)

    if st.button("Generate Report", type="primary"):
        context_reports = []
        retrieved_items = []

        if use_rag:
            with st.spinner(f"Retrieving top-{top_k} similar cases with {retriever_choice}..."):
                retrieved_items = retriever.search("chest x-ray radiology findings", k=top_k)
                for r in retrieved_items:
                    sid = study_id_from_path(r.get("image_path", ""))
                    impression = study_to_impression.get(sid, "")
                    if impression:
                        context_reports.append(impression)

        with st.spinner("MedGemma generating report..."):
            report = generator.generate_report(
                image,
                context_reports=context_reports if use_rag else None,
                structured=structured,
            )

        with col_report:
            st.subheader("Generated Report")
            if structured:
                st.json(generator.parse_structured_report(report))
            else:
                st.markdown(report)

        if use_rag and retrieved_items:
            st.subheader(f"Retrieved Evidence ({retriever_choice})")
            ev_cols = st.columns(len(retrieved_items))
            for col, r, ctx in zip(ev_cols, retrieved_items, context_reports):
                with col:
                    if r.get("image"):
                        col.image(r["image"], use_column_width=True)
                    col.caption(f"Score: {r['score']:.3f}")
                    with col.expander("Retrieved Impression"):
                        st.write(ctx)

# ── Mode: QA ─────────────────────────────────────────────────────────────────
elif mode == "QA Mode":
    st.header("QA Mode")

    EXAMPLE_QUESTIONS = [
        "Is there evidence of pleural effusion?",
        "Does this chest X-ray show signs of cardiomegaly?",
        "Are there any support devices visible?",
        "Is there evidence of atelectasis?",
        "Are the lung fields clear?",
        "Is there a pneumothorax present?",
    ]

    col_q, col_img2 = st.columns([1.2, 1])
    with col_q:
        question = st.text_input(
            "Clinical Question",
            placeholder="Is there evidence of pleural effusion?",
        )
        st.caption("Or choose an example:")
        example_cols = st.columns(3)
        for i, eq in enumerate(EXAMPLE_QUESTIONS[:6]):
            if example_cols[i % 3].button(eq, key=f"eq_{i}"):
                question = eq

    with col_img2:
        st.subheader("Input CXR")
        st.image(image, use_column_width=True)

    if st.button("Get Answer", type="primary") and question:
        context_reports = []
        retrieved_items = []

        with st.spinner(f"Retrieving relevant cases with {retriever_choice}..."):
            retrieved_items = retriever.search(question, k=top_k)
            for r in retrieved_items:
                sid = study_id_from_path(r.get("image_path", ""))
                impression = study_to_impression.get(sid, "")
                if impression:
                    context_reports.append(impression)

        if not context_reports:
            st.warning("No relevant context retrieved. Answer may be less grounded.")
            context_reports = ["No relevant context available."]

        with st.spinner("MedGemma answering..."):
            answer = generator.answer_question(image, question, context_reports)

        st.success(f"**Answer:** {answer}")

        if retrieved_items:
            st.subheader(f"Evidence Used ({retriever_choice})")
            ev_cols = st.columns(len(retrieved_items))
            for col, r, ctx in zip(ev_cols, retrieved_items, context_reports):
                with col:
                    if r.get("image"):
                        col.image(r["image"], use_column_width=True)
                    col.caption(f"Score: {r['score']:.3f}")
                    with col.expander("Supporting Report"):
                        st.write(ctx)
