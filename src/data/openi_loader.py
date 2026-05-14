import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
import glob
from pathlib import Path


class OpenILoader:
    """Loads and parses the OpenI (Indiana University) chest X-ray dataset."""

    COMPARISON_PATTERNS = [
        r'\bcompared to( the)? (previous|prior|old|last)\b.*?[.\n]',
        r'\b(unchanged|stable|no change)\b',
        r'\b(new|interval)\b(?= (development|appearance|finding))',
        r'\bsince( the)? (prior|previous|last)\b.*?[.\n]',
    ]

    def __init__(self, reports_dir: str, images_dir: str):
        self.reports_dir = reports_dir
        self.images_dir = images_dir

    def load(self) -> pd.DataFrame:
        rows = []
        xml_files = glob.glob(os.path.join(self.reports_dir, "*.xml"))
        for xml_path in xml_files:
            row = self._parse_xml(xml_path)
            if row:
                rows.append(row)
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["impression", "image_path"])
        df = df[df["impression"].str.strip().str.len() > 10]
        df = df.reset_index(drop=True)
        return df

    def _parse_xml(self, xml_path: str) -> dict | None:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            return None

        uid = (root.findtext(".//uId") or
               Path(xml_path).stem)
        impression = root.findtext(".//AbstractText[@Label='IMPRESSION']") or ""
        findings = root.findtext(".//AbstractText[@Label='FINDINGS']") or ""

        image_ids = [fig.get("id", "") for fig in root.findall(".//parentImage")]
        frontal_path = self._find_frontal_image(image_ids)

        if not impression.strip() or not frontal_path:
            return None

        return {
            "study_id": uid,
            "impression": self.strip_comparison_language(impression.strip()),
            "findings": self.strip_comparison_language(findings.strip()),
            "image_path": frontal_path,
            "raw_impression": impression.strip(),
        }

    def _find_frontal_image(self, image_ids: list[str]) -> str | None:
        for img_id in image_ids:
            path = os.path.join(self.images_dir, f"{img_id}.png")
            if os.path.exists(path):
                return path
        return None

    def strip_comparison_language(self, text: str) -> str:
        for pattern in self.COMPARISON_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        return text

    def train_val_test_split(
        self, df: pd.DataFrame, train: float = 0.8, val: float = 0.1
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train)
        n_val = int(n * val)
        return (
            df.iloc[:n_train].assign(split="train"),
            df.iloc[n_train:n_train + n_val].assign(split="val"),
            df.iloc[n_train + n_val:].assign(split="test"),
        )
