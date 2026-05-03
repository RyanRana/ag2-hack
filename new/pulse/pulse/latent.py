"""Shared latent state Ψ — the field-wide posterior over plant condition."""

from dataclasses import dataclass, field

import numpy as np


CONDITION_LABELS = [
    "healthy_crop",
    "weed",
    "disease",
    "nutrient_stress",
    "water_stress",
    "pest_damage",
    "ambiguous",
]

INTERVENTION_TYPES = [
    "no_action",
    "laser_zap",            # weed at known location, mechanical
    "targeted_spray",       # herbicide on weed (subject to physics + ecology vetoes)
    "targeted_fungicide",   # disease confirmed
    "targeted_irrigation",  # water stress, specific zone
    "foliar_nutrient",      # nutrient deficiency
    "human_review",         # ambiguous, needs expert eyes
    "rescan_higher_res",    # info-gain says zoom in
]


@dataclass
class PlantInstance:
    """One identified plant in the field image."""

    plant_id: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    crop_image: np.ndarray | None = None

    log_posterior: np.ndarray = field(
        default_factory=lambda: np.full(
            len(CONDITION_LABELS), -np.log(len(CONDITION_LABELS))
        )
    )

    constraint_history: list[str] = field(default_factory=list)

    def posterior(self) -> np.ndarray:
        return np.exp(self.log_posterior - np.logaddexp.reduce(self.log_posterior))

    def top_k(self, k: int = 2) -> list[tuple[str, float]]:
        p = self.posterior()
        idx = np.argsort(p)[::-1][:k]
        return [(CONDITION_LABELS[i], float(p[i])) for i in idx]

    def entropy(self) -> float:
        p = self.posterior()
        return float(-np.sum(p * np.log(p + 1e-12)))


@dataclass
class FieldLatentState:
    """The shared posterior across all plants in the field image."""

    plants: list[PlantInstance] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)
    iteration: int = 0

    def update_plant(
        self,
        plant_id: int,
        log_likelihood_delta: np.ndarray,
        source_agent: str,
    ) -> None:
        """Importance-reweight a single plant's posterior with a new constraint."""
        plant = next(p for p in self.plants if p.plant_id == plant_id)
        plant.log_posterior = plant.log_posterior + log_likelihood_delta
        plant.constraint_history.append(source_agent)

    def disagreement_score(self, plant_id: int, constraints: dict) -> float:
        """Compute pairwise symmetric KL divergence among agent posteriors."""
        plant_constraints = [c for c in constraints.values() if c is not None]
        if len(plant_constraints) < 2:
            return 0.0
        score = 0.0
        n = 0
        for i in range(len(plant_constraints)):
            for j in range(i + 1, len(plant_constraints)):
                pi = self._softmax(plant_constraints[i])
                pj = self._softmax(plant_constraints[j])
                score += np.sum(pi * (np.log(pi + 1e-12) - np.log(pj + 1e-12)))
                score += np.sum(pj * (np.log(pj + 1e-12) - np.log(pi + 1e-12)))
                n += 1
        return float(score / n) if n > 0 else 0.0

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "image_shape": list(self.image_shape),
            "plants": [
                {
                    "plant_id": p.plant_id,
                    "bbox": list(p.bbox),
                    "log_posterior": p.log_posterior.tolist(),
                    "constraint_history": p.constraint_history,
                    "top_k": p.top_k(2),
                    "entropy": p.entropy(),
                }
                for p in self.plants
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldLatentState":
        plants = []
        for p in d.get("plants", []):
            plants.append(
                PlantInstance(
                    plant_id=int(p["plant_id"]),
                    bbox=tuple(int(v) for v in p["bbox"]),  # type: ignore[arg-type]
                    log_posterior=np.asarray(p["log_posterior"], dtype=float),
                    constraint_history=list(p.get("constraint_history", [])),
                )
            )
        shape = d.get("image_shape", (0, 0))
        return cls(
            plants=plants,
            image_shape=tuple(int(v) for v in shape),  # type: ignore[arg-type]
            iteration=int(d.get("iteration", 0)),
        )
