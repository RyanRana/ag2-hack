"""EIGControllerAgent — utility-driven action selection.

Implements the cross-paradigm utility:

    U(action) = E[yield_protected]
              − chem_cost(action)
              − 0.5 × physics_hazard
              − 0.4 × ecological_cost

This is where the three paradigms reconcile into one decision (§9.6.5,
§9.8.5). When physics says spraying is hazardous OR ecology says it
breaks the predator population, the spray utility drops below laser-zap's
and the chemical is refused — even if ML detection was confident.

AG2 idiom: ``register_nested_chats`` spawns one ``ActionEvaluator`` per
``INTERVENTION_TYPES`` entry; the controller fuses their EIGs with physics
+ ecology assessments to pick the argmax.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from autogen import ConversableAgent

from pesto.latent import CONDITION_LABELS, INTERVENTION_TYPES
from pesto.messages import ActionMessage, InterventionAssessmentMessage, TrajectoryMessage


@dataclass
class _UtilityResult:
    action_type: str
    base: float
    hazard_penalty: float
    eco_penalty: float
    total: float


class ActionEvaluator(ConversableAgent):
    """Evaluates the EIG / utility of one candidate action for one plant."""

    def __init__(self, name: str, action_type: str) -> None:
        super().__init__(
            name=name,
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.action_type = action_type
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._compute_eig,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _compute_eig(self, recipient, messages=None, sender=None, config=None):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        log_post = np.asarray(payload.get("log_posterior", []), dtype=float)
        if log_post.size == 0:
            eig = 0.0
        else:
            post = _softmax(log_post)
            eig = _heuristic_eig(post, self.action_type)
        return True, {
            "role": "assistant",
            "name": self.name,
            "content": json.dumps({
                "action_type": self.action_type,
                "eig": float(eig),
                "plant_id": payload.get("plant_id"),
            }),
        }


class EIGControllerAgent(ConversableAgent):
    """Top-level action selector. Composes physics + ecology + ML posterior."""

    def __init__(self, name: str = "controller") -> None:
        super().__init__(
            name=name,
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
        )
        self.evaluators = [
            ActionEvaluator(name=f"eval_{a}", action_type=a) for a in INTERVENTION_TYPES
        ]
        # AG2 idiom 5: register_nested_chats. One nested chat per evaluator.
        self.register_nested_chats(
            chat_queue=[
                {
                    "recipient": ev,
                    "message": _forward_payload,
                    "summary_method": "last_msg",
                    "max_turns": 1,
                }
                for ev in self.evaluators
            ],
            trigger=lambda sender: sender is not self,
        )

    # --- Pure utility math (used in tests + captain) ---------------------

    @staticmethod
    def compute_utility(
        plant_posterior: np.ndarray,
        action_type: str,
        physics_assessment: InterventionAssessmentMessage | None = None,
        ecology_trajectory: TrajectoryMessage | None = None,
    ) -> _UtilityResult:
        i_weed = CONDITION_LABELS.index("weed")
        i_disease = CONDITION_LABELS.index("disease")
        i_water = CONDITION_LABELS.index("water_stress")
        i_nutrient = CONDITION_LABELS.index("nutrient_stress")
        i_pest = CONDITION_LABELS.index("pest_damage")
        i_healthy = CONDITION_LABELS.index("healthy_crop")
        i_ambig = CONDITION_LABELS.index("ambiguous")
        p_weed = float(plant_posterior[i_weed])
        p_disease = float(plant_posterior[i_disease])
        p_water = float(plant_posterior[i_water])
        p_nutrient = float(plant_posterior[i_nutrient])
        p_pest = float(plant_posterior[i_pest])
        p_healthy = float(plant_posterior[i_healthy])
        p_ambig = float(plant_posterior[i_ambig])
        non_healthy_mass = max(0.0, 1.0 - p_healthy)
        # Posterior entropy is high when the agents are uncertain — high entropy
        # makes "review" / "rescan" attractive, but high concentrated probability
        # on an unhealthy class makes the corresponding intervention attractive.
        entropy = float(-np.sum(plant_posterior * np.log(plant_posterior + 1e-12)))
        # ``targeted_fungicide`` doubles as the chemical-treatment slot for both
        # disease and pest_damage — pest detection should NEVER fall through to
        # no_action. The friendly UI label resolves to "treat with insecticide"
        # when the dominant condition is pest_damage.
        base = {
            "no_action": p_healthy * 1.0 - 0.55 * non_healthy_mass,
            "laser_zap": p_weed * 0.95 - 0.04 + p_pest * 0.30,
            "targeted_spray": p_weed * 0.95 - 0.08,
            "targeted_fungicide": (p_disease + p_pest) * 0.92 - 0.06,
            "targeted_irrigation": p_water * 0.90 - 0.03,
            "foliar_nutrient": p_nutrient * 0.85 - 0.04,
            "human_review": entropy * 0.40 + p_ambig * 0.50 - 0.05,
            "rescan_higher_res": entropy * 0.28 + p_ambig * 0.40 - 0.02,
        }.get(action_type, 0.0)
        hazard_penalty = 0.5 * physics_assessment.hazard_score if physics_assessment else 0.0
        eco_penalty = 0.4 * ecology_trajectory.ecological_cost_score if ecology_trajectory else 0.0
        total = float(base - hazard_penalty - eco_penalty)
        return _UtilityResult(
            action_type=action_type,
            base=float(base),
            hazard_penalty=float(hazard_penalty),
            eco_penalty=float(eco_penalty),
            total=total,
        )

    @classmethod
    def select_action(
        cls,
        plant_id: int,
        plant_posterior: np.ndarray,
        physics_per_action: dict[str, InterventionAssessmentMessage],
        ecology_per_action: dict[str, TrajectoryMessage],
        interventions: list[str] | None = None,
    ) -> ActionMessage:
        """Pick argmax utility over the action space.

        Pass ``interventions`` to restrict the candidate set — the SDK uses
        this to toggle interventions off without retraining.
        """
        action_space = list(interventions) if interventions else INTERVENTION_TYPES
        results: list[_UtilityResult] = []
        for action in action_space:
            phy = physics_per_action.get(action)
            eco = ecology_per_action.get(action)
            results.append(cls.compute_utility(plant_posterior, action, phy, eco))
        best = max(results, key=lambda r: r.total)
        # Expected information gain proxy: posterior entropy reduction estimate.
        eig = _heuristic_eig(plant_posterior, best.action_type)
        return ActionMessage(
            sender="controller",
            timestamp=time.time(),
            plant_id=int(plant_id),
            action_type=best.action_type,
            action_params={},
            expected_information_gain=float(eig),
            expected_utility=float(best.total),
            physics_hazard_score=float(physics_per_action[best.action_type].hazard_score)
                if best.action_type in physics_per_action
                else 0.0,
        )


# --- Helpers --------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def _heuristic_eig(post: np.ndarray, action_type: str) -> float:
    entropy = float(-np.sum(post * np.log(post + 1e-12)))
    weights = {
        "laser_zap": post[CONDITION_LABELS.index("weed")] * 0.9,
        "targeted_spray": post[CONDITION_LABELS.index("weed")] * 0.9,
        "targeted_fungicide": post[CONDITION_LABELS.index("disease")] * 0.85,
        "targeted_irrigation": post[CONDITION_LABELS.index("water_stress")] * 0.85,
        "foliar_nutrient": post[CONDITION_LABELS.index("nutrient_stress")] * 0.85,
        "human_review": entropy * 0.7,
        "rescan_higher_res": entropy * 0.5,
        "no_action": post[CONDITION_LABELS.index("healthy_crop")] * 1.0,
    }
    return float(weights.get(action_type, 0.0))


def _forward_payload(recipient, messages, sender):
    latest = messages[-1] if messages else {}
    return {"content": latest.get("content", "{}"), "name": "controller"}
