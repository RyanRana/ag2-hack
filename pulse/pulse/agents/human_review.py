"""HumanReviewProxy — UserProxyAgent with conditional human input (§8.6)."""

from __future__ import annotations

from autogen import UserProxyAgent


class HumanReviewProxy(UserProxyAgent):
    """Human-in-the-loop only when the system can't resolve ambiguity."""

    def __init__(self) -> None:
        super().__init__(
            name="human_reviewer",
            human_input_mode="TERMINATE",
            code_execution_config=False,
            llm_config=False,
            is_termination_msg=lambda msg: True,
        )
