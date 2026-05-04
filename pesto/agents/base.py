"""ChannelAgent — code-only AG2 agent that emits ConstraintMessages.

The five ML diagnosis agents, the physics agent, the biophysics agent, and
the ecology agent all subclass this — same envelope, four different
inference paradigms (the architectural moment §1 talks about).

AG2 idiom: ConversableAgent with llm_config=False + register_reply at
position 0 with remove_other_reply_funcs=True so neither LLM, code, nor
function defaults fire.
"""

from __future__ import annotations

import json

from autogen import ConversableAgent

from pesto.latent import FieldLatentState
from pesto.messages import ConstraintMessage


class ChannelAgent(ConversableAgent):
    """Base class for code-only model agents in the inference loop."""

    def __init__(self, name: str, **kwargs: object) -> None:
        super().__init__(
            name=name,
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False,
            **kwargs,
        )
        self.register_reply(
            trigger=[ConversableAgent, None],
            reply_func=self._emit_constraint_reply,
            position=0,
            remove_other_reply_funcs=True,
        )

    def _emit_constraint_reply(
        self,
        recipient,
        messages=None,
        sender=None,
        config=None,
    ):
        latest = messages[-1] if messages else {}
        payload = json.loads(latest.get("content", "{}"))
        latent = FieldLatentState.from_dict(payload["latent"])
        image_path = payload.get("image_path")
        constraint = self.emit_constraint(image_path, latent)
        return True, {
            "role": "assistant",
            "name": self.name,
            "content": json.dumps(self._serialize_constraint(constraint)),
        }

    @staticmethod
    def _serialize_constraint(c: ConstraintMessage) -> dict:
        return {
            "sender": c.sender,
            "timestamp": c.timestamp,
            "iteration": c.iteration,
            "per_plant_log_likelihoods": {
                int(k): v.tolist() for k, v in c.per_plant_log_likelihoods.items()
            },
            "per_plant_residual": {int(k): float(v) for k, v in c.per_plant_residual.items()},
            "per_plant_confidence": {int(k): float(v) for k, v in c.per_plant_confidence.items()},
            "labels_discriminated": c.labels_discriminated,
            "metadata": c.metadata,
        }

    def emit_constraint(
        self, image_path: str | None, latent: FieldLatentState
    ) -> ConstraintMessage:
        raise NotImplementedError
