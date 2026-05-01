import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GoToSleep(Tool):
    """Stop listening and wait for the wake word before responding again."""

    name = "go_to_sleep"
    description = (
        "Stop listening and go to sleep. The mic input will be ignored until the user "
        "says the wake word again. Use this when the user explicitly asks to be left alone, "
        "to stop the conversation, or says good night / goodbye."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional reason for going to sleep (e.g., 'user said goodnight', 'user asked for silence').",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Put the wake-word gate to sleep."""
        reason = kwargs.get("reason", "tool call")
        logger.info("Tool call: go_to_sleep reason=%s", reason)

        if deps.wake_word_gate is None:
            return {"status": "wake-word gate disabled", "reason": reason}

        deps.wake_word_gate.sleep(reason=reason)
        return {
            "status": "asleep",
            "wakeword": deps.wake_word_gate.wakeword_name,
            "reason": reason,
        }
