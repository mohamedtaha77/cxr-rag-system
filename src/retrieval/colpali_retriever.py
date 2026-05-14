import os
import pickle
from pathlib import Path
from PIL import Image


class ColPaliRetriever:
    """
    Multimodal retrieval using ColPali (late-interaction over image patches).
    Wraps the Byaldi library for a familiar RAGatouille-style API.
    Bypasses OCR entirely — queries raw images directly.
    """

    INDEX_NAME = "openi_colpali_index"
    MODEL_NAME = "vidore/colpali-v1.3"

    def __init__(self):
        # Import here so non-Colab environments can import the module without crashing
        from byaldi import RAGMultiModalModel
        self.model = RAGMultiModalModel.from_pretrained(self.MODEL_NAME)
        self._image_path_map: dict[int, str] = {}  # doc_id → file path

    def build_index(self, images_dir: str, index_save_dir: str | None = None):
        """Index all PNG/JPG images in images_dir. Saves index alongside images by default."""
        images_dir = str(images_dir)
        self.model.index(
            input_path=images_dir,
            index_name=self.INDEX_NAME,
            store_collection_with_index=True,
            overwrite=True,
        )
        # Build doc_id → path mapping for later image retrieval
        self._build_path_map(images_dir)
        if index_save_dir:
            self._save_path_map(index_save_dir)

    def _build_path_map(self, images_dir: str):
        paths = sorted(
            p for p in Path(images_dir).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        self._image_path_map = {i: str(p) for i, p in enumerate(paths)}

    def _save_path_map(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "colpali_path_map.pkl"), "wb") as f:
            pickle.dump(self._image_path_map, f)

    def load_path_map(self, save_dir: str):
        with open(os.path.join(save_dir, "colpali_path_map.pkl"), "rb") as f:
            self._image_path_map = pickle.load(f)

    @classmethod
    def from_index(cls, index_dir: str) -> "ColPaliRetriever":
        """Load a previously saved index without rebuilding."""
        from byaldi import RAGMultiModalModel
        instance = cls.__new__(cls)
        instance.model = RAGMultiModalModel.from_index(
            index_path=index_dir,
            index_name=cls.INDEX_NAME,
        )
        instance._image_path_map = {}
        return instance

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Search the index with a text query.
        Returns list of {doc_id, score, image_path, image}.
        """
        results = self.model.search(query, k=k)
        output = []
        for r in results:
            doc_id = r.doc_id
            image_path = self._image_path_map.get(doc_id, "")
            pil_image = None
            if image_path and os.path.exists(image_path):
                pil_image = Image.open(image_path).convert("RGB")
            output.append({
                "doc_id": doc_id,
                "score": r.score,
                "image_path": image_path,
                "image": pil_image,
            })
        return output
