"""
Model and index loading utilities.
All loaders are cached by Streamlit so models are loaded only once per session.
"""
import os
import streamlit as st


INDEX_DIR = os.environ.get("INDEX_DIR", "data/indices")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")


@st.cache_resource(show_spinner="Loading ColPali index...")
def load_colpali():
    from src.retrieval.colpali_retriever import ColPaliRetriever
    retriever = ColPaliRetriever.from_index(os.path.join(INDEX_DIR, "colpali_index"))
    retriever.load_path_map(os.path.join(INDEX_DIR, "colpali_index"))
    return retriever


@st.cache_resource(show_spinner="Loading CLIP index...")
def load_clip():
    from src.retrieval.clip_retriever import CLIPRetriever
    retriever = CLIPRetriever()
    retriever.load_index(os.path.join(INDEX_DIR, "clip_index"))
    return retriever


@st.cache_resource(show_spinner="Loading MedGemma (4-bit, ~3 GB VRAM)...")
def load_medgemma():
    from src.generation.medgemma_generator import MedGemmaGenerator
    return MedGemmaGenerator(hf_token=HF_TOKEN, load_in_4bit=True)


@st.cache_data(show_spinner=False)
def load_corpus():
    import pandas as pd
    corpus_path = os.environ.get("CORPUS_PATH", "data/processed/reports_corpus.csv")
    df = pd.read_csv(corpus_path)
    return dict(zip(df["study_id"], df["impression"]))


def get_retriever(choice: str):
    if "ColPali" in choice:
        return load_colpali()
    return load_clip()


def study_id_from_path(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]
