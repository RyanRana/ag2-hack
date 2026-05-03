"""End-to-end runtime tests — the three paradigm-veto moments (§13 Phase 6).

We use injected ML agents (deterministic constraints) and the real physics
+ biophysics + ecology agents. No LLM key needed.
"""

from __future__ import annotations

import time

import numpy as np

from pulse.agents.controller import EIGControllerAgent
from pulse.agents.ecological_dynamics import EcologicalDynamicsAgent
from pulse.agents.pesticide_fate import PesticideFateAgent
from pulse.agents.water_balance import WaterBalanceAgent
from pulse.captain import PulseCaptain
from pulse.latent import CONDITION_LABELS, FieldLatentState, INTERVENTION_TYPES, PlantInstance
from pulse.messages import ConstraintMessage


# --- Stub ML agents -------------------------------------------------------


class _StubMLAgent:
    """Emits a fixed ConstraintMessage. Used to set up controlled scenarios."""

    def __init__(self, name: str, log_likelihoods: dict[int, list[float]]) -> None:
        self.name = name
        self._lls = log_likelihoods

    def emit_constraint(self, image_path, latent) -> ConstraintMessage:
        per_ll = {pid: np.asarray(v, dtype=float) for pid, v in self._lls.items()}
        return ConstraintMessage(
            sender=self.name,
            timestamp=time.time(),
            iteration=latent.iteration,
            per_plant_log_likelihoods=per_ll,
            per_plant_residual={pid: 0.0 for pid in per_ll},
            per_plant_confidence={pid: 0.8 for pid in per_ll},
            labels_discriminated=list(CONDITION_LABELS),
        )


def _ll_for(label: str, mass: float = 4.0) -> list[float]:
    """Return a log-likelihood vector concentrated on one label."""
    out = [0.0] * len(CONDITION_LABELS)
    out[CONDITION_LABELS.index(label)] = mass
    return out


# --- Controller utility math ---------------------------------------------


def test_utility_baseline_disease_picks_fungicide():
    """No penalties: with confident disease detection, fungicide beats laser/spray."""
    posterior = np.zeros(len(CONDITION_LABELS))
    posterior[CONDITION_LABELS.index("disease")] = 0.95
    posterior[CONDITION_LABELS.index("ambiguous")] = 0.05
    u_fungicide = EIGControllerAgent.compute_utility(posterior, "targeted_fungicide")
    u_laser = EIGControllerAgent.compute_utility(posterior, "laser_zap")
    u_no = EIGControllerAgent.compute_utility(posterior, "no_action")
    assert u_fungicide.total > u_laser.total
    assert u_fungicide.total > u_no.total


def test_physics_hazard_flips_spray_to_laser():
    """High physics hazard penalises spray below laser_zap utility."""
    posterior = np.zeros(len(CONDITION_LABELS))
    posterior[CONDITION_LABELS.index("weed")] = 0.95
    posterior[CONDITION_LABELS.index("ambiguous")] = 0.05

    physics = PesticideFateAgent()
    field = {
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0]},
            {"plant_id": 1, "xy_m": [0.2, 0.0]},  # 20cm downwind
        ],
        "wind_dir_deg": 270,
        "wind_speed_m_s": 1.5,
    }
    spray_phy = physics.assess_intervention(
        0, "targeted_spray",
        {"volume_ml": 5.0, "concentration_g_l": 360.0, "application_duration_s": 0.3},
        field, "atrazine",
    )
    assert spray_phy.hazard_score > 0.4

    laser_phy = physics.assess_intervention(0, "laser_zap", {}, field, "atrazine")
    u_spray = EIGControllerAgent.compute_utility(
        posterior, "targeted_spray", physics_assessment=spray_phy
    )
    u_laser = EIGControllerAgent.compute_utility(
        posterior, "laser_zap", physics_assessment=laser_phy
    )
    assert u_laser.total > u_spray.total


def test_ecology_cost_flips_spray_to_laser():
    """High ecology cost (chlorpyrifos) flips spray → laser."""
    posterior = np.zeros(len(CONDITION_LABELS))
    posterior[CONDITION_LABELS.index("weed")] = 0.95
    posterior[CONDITION_LABELS.index("ambiguous")] = 0.05

    eco = EcologicalDynamicsAgent()
    spray_eco = eco.assess_intervention({
        "plant_id": 0,
        "action_type": "targeted_spray",
        "action_params": {"volume_ml": 5.0, "concentration_g_l": 360.0},
        "chemistry": "chlorpyrifos",
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    })
    laser_eco = eco.assess_intervention({
        "plant_id": 0,
        "action_type": "laser_zap",
        "action_params": {},
        "chemistry": None,
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    })
    assert spray_eco.ecological_cost_score > 0.7
    assert laser_eco.ecological_cost_score < 0.1

    u_spray = EIGControllerAgent.compute_utility(
        posterior, "targeted_spray", ecology_trajectory=spray_eco
    )
    u_laser = EIGControllerAgent.compute_utility(
        posterior, "laser_zap", ecology_trajectory=laser_eco
    )
    assert u_laser.total > u_spray.total


def test_water_balance_resolves_disease_water_disagreement():
    """ML disease/water-stress disagreement resolved by biophysics."""
    posterior_log = np.zeros(len(CONDITION_LABELS))
    # Two ML agents disagree.
    posterior_log[CONDITION_LABELS.index("disease")] = 2.0
    posterior_log[CONDITION_LABELS.index("water_stress")] = 2.0
    # Add posterior in linear space.
    plant = PlantInstance(plant_id=0, bbox=(0, 0, 10, 10))
    plant.log_posterior = posterior_log.copy()
    pre = plant.posterior()
    assert abs(pre[CONDITION_LABELS.index("disease")]
               - pre[CONDITION_LABELS.index("water_stress")]) < 1e-6

    # Biophysics says the field is well-watered → suppresses water_stress.
    wb = WaterBalanceAgent()
    msg = wb.emit_constraint({
        "latent": {"iteration": 0},
        "field_state": {
            "soil_moisture_m3_m3": 0.32,
            "soil_texture": "loam",
            "T_C": 22.0, "RH_pct": 60, "u2_m_s": 1.5, "R_n_MJ_m2_d": 14.0,
            "crop_type": "tomato",
            "plants": [{"plant_id": 0}],
        },
    })
    plant.log_posterior += msg.per_plant_log_likelihoods[0]
    post = plant.posterior()
    # Disease should now dominate.
    assert post[CONDITION_LABELS.index("disease")] > post[CONDITION_LABELS.index("water_stress")]


# --- Full captain pipeline ------------------------------------------------


def _make_latent(plants_bboxes):
    f = FieldLatentState(image_shape=(480, 640))
    for i, b in enumerate(plants_bboxes):
        f.plants.append(PlantInstance(plant_id=i, bbox=b))
    return f


class _PreSeededCaptain(PulseCaptain):
    """Captain wired to a pre-seeded latent (no real YOLO needed)."""

    def __init__(self, *, ml_agents, latent: FieldLatentState, **kwargs) -> None:
        super().__init__(ml_agents=ml_agents, **kwargs)
        self._seed_latent = latent

    def _initialise_latent(self, image_path: str) -> FieldLatentState:
        return self._seed_latent


def test_captain_full_pipeline_three_paradigm_moments():
    """Captain run produces ActionMessages reflecting the three paradigm moments."""
    latent = _make_latent([(10, 10, 100, 100), (200, 200, 300, 300), (350, 350, 450, 450)])
    # Plant 0: ML says weed (high confidence) — physics will veto due to neighbor 1
    #          and ecology will veto with chlorpyrifos.
    # Plant 1: ML disagrees disease vs water — biophysics resolves to disease.
    # Plant 2: healthy.
    ml_a = _StubMLAgent("weed_detector", {
        0: _ll_for("weed", 4.0),
        1: _ll_for("disease", 0.5),
        2: _ll_for("healthy_crop", 2.0),
    })
    ml_b = _StubMLAgent("disease_classifier", {
        0: _ll_for("weed", 1.0),
        1: _ll_for("disease", 3.0),
        2: _ll_for("healthy_crop", 2.0),
    })
    ml_c = _StubMLAgent("segmentation", {
        0: _ll_for("weed", 1.0),
        1: _ll_for("water_stress", 3.0),  # disagrees with disease_classifier
        2: _ll_for("healthy_crop", 2.0),
    })

    captain = _PreSeededCaptain(
        ml_agents=[ml_a, ml_b, ml_c],
        latent=latent,
    )
    field_state = {
        # Place plant 1 just downwind of plant 0 (20 cm) so spray hazards trigger.
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0]},
            {"plant_id": 1, "xy_m": [0.2, 0.0]},
            {"plant_id": 2, "xy_m": [-1.0, 0.5]},
        ],
        "wind_dir_deg": 270,
        "wind_speed_m_s": 1.5,
        "soil_moisture_m3_m3": 0.32,
        "soil_texture": "loam",
        "T_C": 22.0, "RH_pct": 60, "u2_m_s": 1.5, "R_n_MJ_m2_d": 14.0,
        "crop_type": "tomato",
        "initial_populations": {"pest": 100.0, "predator": 20.0, "parasitoid": 15.0},
    }
    chemistry = {0: "chlorpyrifos", 1: "glyphosate", 2: "glyphosate"}

    out = captain.run_inference("dummy.jpg", field_state, chemistry_per_plant=chemistry)
    actions = {a["plant_id"]: a for a in out["actions"]}
    # Plant 0: ML+ecology+physics all push toward laser, not spray.
    assert actions[0]["action_type"] in {"laser_zap", "no_action"}
    assert actions[0]["action_type"] != "targeted_spray"
    # Plant 1: water_stress was suppressed by biophysics, disease wins.
    posterior_p1 = np.exp(np.asarray(out["latent"]["plants"][1]["log_posterior"]))
    posterior_p1 = posterior_p1 / posterior_p1.sum()
    i_disease = CONDITION_LABELS.index("disease")
    i_water = CONDITION_LABELS.index("water_stress")
    assert posterior_p1[i_disease] > posterior_p1[i_water]
    # Plant 2: healthy → no_action.
    assert actions[2]["action_type"] == "no_action"


def test_captain_pipeline_emits_paradigm_specific_events():
    """The captain emits typed events for each paradigm — required for the dashboard."""
    latent = _make_latent([(10, 10, 100, 100), (200, 200, 300, 300)])
    ml_a = _StubMLAgent("weed_detector", {
        0: _ll_for("weed", 3.0), 1: _ll_for("healthy_crop", 2.0)
    })
    ml_b = _StubMLAgent("health_classifier", {
        0: _ll_for("weed", 2.0), 1: _ll_for("healthy_crop", 2.0)
    })

    events: list[tuple[str, object]] = []

    captain = _PreSeededCaptain(
        ml_agents=[ml_a, ml_b],
        latent=latent,
        emit_event=lambda kind, payload: events.append((kind, payload)),
    )
    field_state = {
        "plants": [
            {"plant_id": 0, "xy_m": [0.0, 0.0]},
            {"plant_id": 1, "xy_m": [0.4, 0.0]},
        ],
        "wind_dir_deg": 270, "wind_speed_m_s": 2.0,
        "soil_moisture_m3_m3": 0.30,
        "soil_texture": "loam",
        "T_C": 22.0, "RH_pct": 55, "u2_m_s": 1.5, "R_n_MJ_m2_d": 14.0,
        "crop_type": "tomato",
    }
    captain.run_inference("dummy.jpg", field_state)

    kinds = {k for k, _ in events}
    assert {"constraint", "physics_assessment", "ecology_trajectory", "action"} <= kinds
    # At least one chlorpyrifos-class ecology event with eco_cost > 0.5 — only
    # if the user requested it. Default chemistry is glyphosate so eco_cost
    # is low; just verify the event shape.
    eco_events = [p for k, p in events if k == "ecology_trajectory"]
    assert all("ecological_cost_score" in p for p in eco_events)
    assert all("predator_drop_pct_d14" in p["cost_breakdown"] for p in eco_events)
