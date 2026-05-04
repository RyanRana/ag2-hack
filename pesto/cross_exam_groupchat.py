"""GroupChat with custom speaker_selection_method (AG2 idiom 3, §8.2)."""

from __future__ import annotations

import numpy as np
from autogen import GroupChat

from pesto.messages import ConstraintMessage


class CrossExamGroupChat(GroupChat):
    """Speaker selection picks the agent most likely to contest the consensus."""

    def __init__(self, agents, latent, max_round: int = 8) -> None:
        super().__init__(
            agents=agents,
            messages=[],
            max_round=max_round,
            speaker_selection_method=self._select_most_contested,
        )
        self.latent = latent
        self._latest_constraints: dict[str, ConstraintMessage] = {}

    def record_constraint(self, agent_name: str, constraint: ConstraintMessage) -> None:
        self._latest_constraints[agent_name] = constraint

    def _select_most_contested(self, last_speaker, groupchat):
        if len(self._latest_constraints) < 2:
            idx = (groupchat.agents.index(last_speaker) + 1) % len(groupchat.agents)
            return groupchat.agents[idx]
        spoken = set(self._latest_constraints.keys())
        remaining = [a for a in groupchat.agents if a.name not in spoken]
        if remaining:
            return remaining[0]
        # Everyone has spoken — pick the agent furthest from the consensus.
        all_lls = list(self._latest_constraints.values())
        consensus_per_plant: dict[int, np.ndarray] = {}
        for plant_id in all_lls[0].per_plant_log_likelihoods:
            stack = np.stack([
                np.asarray(c.per_plant_log_likelihoods[plant_id])
                for c in all_lls
                if plant_id in c.per_plant_log_likelihoods
            ])
            consensus_per_plant[plant_id] = stack.mean(axis=0)
        scores: dict[str, float] = {}
        for name, c in self._latest_constraints.items():
            divergence = 0.0
            for plant_id, ll in c.per_plant_log_likelihoods.items():
                ll_vec = np.asarray(ll)
                consensus = consensus_per_plant.get(plant_id)
                if consensus is None:
                    continue
                divergence += float(np.linalg.norm(ll_vec - consensus))
            scores[name] = divergence
        # Pick the most divergent that isn't the last speaker.
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        for name, _ in ranked:
            if name != getattr(last_speaker, "name", None):
                for a in groupchat.agents:
                    if a.name == name:
                        return a
        idx = (groupchat.agents.index(last_speaker) + 1) % len(groupchat.agents)
        return groupchat.agents[idx]
