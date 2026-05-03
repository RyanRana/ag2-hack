"""RAG retriever — FAISS + sentence-transformers for agronomic knowledge.

Provides local vector search over agronomic literature (extension service
bulletins, IPM guidelines, treatment thresholds). The controller queries
with (disease, crop, region, season) and receives treatment context that
adjusts chemical choice + treatment thresholds.

All local — no external API. FAISS in-process, sentence-transformers
embedding model cached to disk after first download.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 (~80MB, CPU).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


_INDEX_DIR = Path(__file__).parent.parent.parent / "data" / "rag_index"

# Built-in agronomic knowledge base (can be extended with user documents)
_DEFAULT_DOCUMENTS = [
    {
        "id": "early_blight_tomato",
        "text": "Early blight (Alternaria solani) on tomato. Threshold: 2 lesions per "
                "leaf before treatment is justified. First-line treatment: chlorothalonil. "
                "Resistance to mancozeb documented in central regions since 2022. "
                "Apply preventively when humidity >80% for 3+ days.",
        "tags": ["disease", "tomato", "fungal", "early_blight"],
    },
    {
        "id": "late_blight_potato",
        "text": "Late blight (Phytophthora infestans) on potato/tomato. Highly aggressive "
                "under cool wet conditions (15-22C, >90% humidity). Treatment: metalaxyl + "
                "mancozeb. Begin applications at first sign or when BLITECAST forecast "
                "indicates 18+ severity values. 5-7 day spray interval during active spread.",
        "tags": ["disease", "potato", "tomato", "fungal", "late_blight"],
    },
    {
        "id": "powdery_mildew_cucurbits",
        "text": "Powdery mildew on cucurbits. Threshold: 1 leaf in 50 with visible "
                "colonies. Treatment: sulfur (organic) or myclobutanil. Hot dry weather "
                "favors development (unlike most fungal diseases). Resistant varieties "
                "available for cucumber and melon.",
        "tags": ["disease", "cucurbits", "fungal", "powdery_mildew"],
    },
    {
        "id": "aphid_management",
        "text": "Aphid management in vegetable crops. Economic threshold: 50+ aphids per "
                "plant or 20% of plants infested. Biological control: release Aphidius "
                "colemani at 0.5/m2 weekly. Chemical: imidacloprid (systemic) or pyrethrins "
                "(contact). Avoid broad-spectrum insecticides that kill natural enemies. "
                "Scout weekly during vegetative growth.",
        "tags": ["pest", "aphid", "vegetables", "biological_control"],
    },
    {
        "id": "water_stress_indicators",
        "text": "Water stress identification. Visual cues: leaf rolling (grasses), "
                "wilting (broadleaves), grey-green color shift, reduced leaf expansion. "
                "Distinguish from disease: water stress affects entire canopy uniformly, "
                "disease typically starts focal. Soil moisture sensor threshold: "
                "irrigate when below 50% available water capacity.",
        "tags": ["water_stress", "irrigation", "diagnosis"],
    },
    {
        "id": "nitrogen_deficiency",
        "text": "Nitrogen deficiency in crops. Lower leaves yellow first (mobile nutrient). "
                "Distinguish from iron deficiency (interveinal chlorosis on new leaves) and "
                "sulfur deficiency (uniform chlorosis on new leaves). Foliar urea spray "
                "(2-3% solution) for quick correction. Soil test: <20 ppm NO3-N suggests "
                "deficiency in most crops.",
        "tags": ["nutrient_stress", "nitrogen", "diagnosis"],
    },
    {
        "id": "ipm_spray_timing",
        "text": "Integrated Pest Management spray timing guidelines. Apply pesticides "
                "in early morning or late evening to minimize drift and pollinator exposure. "
                "Wind speed must be <10 mph (4.5 m/s). Temperature <85F (29C) for most "
                "products. Buffer zones: 30m from water bodies, 15m from hedgerows. "
                "Record all applications for compliance.",
        "tags": ["spray", "timing", "ipm", "safety"],
    },
    {
        "id": "organic_alternatives",
        "text": "Organic pest management alternatives. Bt (Bacillus thuringiensis) for "
                "caterpillars: apply to young larvae, reapply after rain. Neem oil for "
                "soft-bodied insects: effective against aphids, whiteflies, mites. "
                "Copper-based fungicides for bacterial/fungal diseases: phytotoxic above "
                "90F. Sulfur for powdery mildew: do not mix with oil sprays.",
        "tags": ["organic", "biological_control", "fungicide", "insecticide"],
    },
    {
        "id": "herbicide_resistance",
        "text": "Herbicide resistance management. Rotate mode-of-action groups annually. "
                "Glyphosate-resistant weeds documented in 40+ species globally. Integrate "
                "mechanical control (cultivation, mowing) with chemical. Cover crops "
                "suppress weed seedbanks. Scout pre-emergence to select appropriate "
                "pre-emergent herbicide based on weed species present.",
        "tags": ["weed", "herbicide", "resistance", "management"],
    },
    {
        "id": "seedling_disease_prevention",
        "text": "Seedling disease prevention (damping off). Caused by Pythium, "
                "Rhizoctonia, Fusarium. Use treated seed, well-drained media, avoid "
                "overwatering. Fungicide drench: mefenoxam for Pythium, fludioxonil for "
                "Rhizoctonia. Seedlings are most vulnerable in first 2 weeks. Maintain "
                "soil temperature 65-75F for optimal germination.",
        "tags": ["seedling", "disease", "prevention", "fungal"],
    },
]


class AgronomicRAG:
    """Vector search over agronomic knowledge base.

    Uses sentence-transformers for embedding and FAISS for nearest-neighbor
    search. The index is built from a knowledge base of agronomic documents.
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        embedding_model: Any | None = None,
        index: Any | None = None,
    ) -> None:
        self._documents = documents or list(_DEFAULT_DOCUMENTS)
        self._embedding_model = embedding_model
        self._index = index
        self._embeddings: np.ndarray | None = None
        self._built = index is not None

    @property
    def is_built(self) -> bool:
        return self._built

    def _load_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        from sentence_transformers import SentenceTransformer
        self._embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._embedding_model

    def build_index(self) -> None:
        """Build the FAISS index from the document collection."""
        import faiss

        model = self._load_embedding_model()
        texts = [doc["text"] for doc in self._documents]
        embeddings = model.encode(texts, normalize_embeddings=True)
        self._embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # inner product (cosine with normalized vecs)
        self._index.add(self._embeddings)
        self._built = True

    def query(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> list[dict]:
        """Search for relevant agronomic documents.

        Parameters
        ----------
        query_text : Natural language query, e.g. "early blight tomato treatment".
        top_k : Number of results to return.

        Returns
        -------
        List of dicts with "id", "text", "tags", "score" keys, sorted by relevance.
        """
        if not self._built:
            self.build_index()

        model = self._load_embedding_model()
        query_emb = model.encode([query_text], normalize_embeddings=True)
        query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)

        scores, indices = self._index.search(query_emb, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def query_for_treatment(
        self,
        condition: str,
        crop: str = "",
        season: str = "",
        region: str = "",
    ) -> list[dict]:
        """Structured query for treatment recommendations.

        Parameters
        ----------
        condition : Detected condition (e.g. "disease", "pest_damage").
        crop : Crop type if known.
        season : Season or month.
        region : Geographic region.

        Returns
        -------
        Top-3 relevant documents with treatment context.
        """
        parts = [condition]
        if crop:
            parts.append(crop)
        if season:
            parts.append(season)
        if region:
            parts.append(region)
        query = " ".join(parts) + " treatment management"
        return self.query(query, top_k=3)

    def add_documents(self, documents: list[dict]) -> None:
        """Add new documents to the knowledge base and rebuild the index."""
        self._documents.extend(documents)
        self._built = False

    def save_index(self, directory: Path | None = None) -> Path:
        """Persist the FAISS index and document metadata to disk."""
        import faiss

        d = directory or _INDEX_DIR
        d.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(d / "index.faiss"))
        # Save documents as JSON
        with open(d / "documents.json", "w") as f:
            json.dump(self._documents, f, indent=2)
        return d

    @classmethod
    def load_index(cls, directory: Path | None = None) -> "AgronomicRAG":
        """Load a persisted index from disk."""
        import faiss

        d = directory or _INDEX_DIR
        index_path = d / "index.faiss"
        docs_path = d / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            return cls()

        index = faiss.read_index(str(index_path))
        with open(docs_path) as f:
            documents = json.load(f)

        rag = cls(documents=documents, index=index)
        rag._built = True
        return rag


def compute_rag_adjustment(
    retrieved_docs: list[dict],
    action_type: str,
) -> float:
    """Compute a utility adjustment from RAG-retrieved treatment context.

    Positive adjustment = RAG supports this action.
    Negative adjustment = RAG advises against or suggests alternatives.

    Parameters
    ----------
    retrieved_docs : Documents from AgronomicRAG.query_for_treatment().
    action_type : The candidate intervention type.

    Returns
    -------
    Utility adjustment in range [-0.3, +0.3].
    """
    if not retrieved_docs:
        return 0.0

    adjustment = 0.0
    top = retrieved_docs[0]
    text = top.get("text", "").lower()
    score = top.get("score", 0.0)

    # Only apply meaningful adjustments for high-confidence retrievals
    if score < 0.3:
        return 0.0

    # If RAG recommends treatment and action is treatment-type
    treatment_actions = {"targeted_spray", "targeted_fungicide", "foliar_nutrient"}
    if action_type in treatment_actions:
        if "resistance" in text or "resistant" in text:
            adjustment -= 0.15  # resistance documented → penalise default chemical
        if "threshold" in text:
            adjustment += 0.1  # threshold guidance available → supports informed treatment
        if "organic" in text or "biological" in text:
            adjustment += 0.05  # organic alternative available

    # If action is no_action but RAG says condition is serious
    if action_type == "no_action":
        if "aggressive" in text or "highly" in text:
            adjustment -= 0.2  # serious condition → don't do nothing

    # If action is human_review and RAG has relevant info
    if action_type == "human_review":
        if score > 0.5:
            adjustment -= 0.1  # good RAG match → maybe can act without review

    return float(np.clip(adjustment, -0.3, 0.3))
