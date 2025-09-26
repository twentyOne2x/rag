import copy
import json
import logging
import os
from typing import Optional, List, Tuple, cast, Union
from datetime import datetime

# ReAct agent (core, 0.10.15) with legacy fallback
try:
    from llama_index.core.agent.workflow import ReActAgent
except ImportError:
    from llama_index.legacy.agent.legacy.react.base import ReActAgent  # type: ignore

# Reasoning step types (core first, legacy fallback)
try:
    from llama_index.core.agent.react.types import (
        BaseReasoningStep,
        ActionReasoningStep,
        ObservationReasoningStep,
        ResponseReasoningStep,
    )
except ImportError:
    from llama_index.legacy.agent.react.types import (  # type: ignore
        BaseReasoningStep,
        ActionReasoningStep,
        ObservationReasoningStep,
        ResponseReasoningStep,
    )

# Callbacks & tracing
from llama_index.core.callbacks import trace_method, CBEventType, EventPayload

# Chat response types (core first, legacy fallback)
try:
    from llama_index.core.chat_engine.types import AgentChatResponse
except ImportError:
    class AgentChatResponse:  # type: ignore
        def __init__(self, response: str, sources: Optional[List[str]] = None):
            self.response = response
            self.sources = sources or []

try:
    from llama_index.core.llms import ChatMessage, MessageRole, ChatResponse
except ImportError:
    from llama_index.core.base.llms.types import (  # type: ignore
        ChatMessage,
        MessageRole,
        ChatResponse,
    )

from llama_index.core.utils import print_text

from src.Llama_index_sandbox.custom_react_agent.callbacks.schema import ExtendedEventPayload
from src.Llama_index_sandbox.custom_react_agent.tools.query_engine_prompts import AVOID_CITING_CONTEXT
from src.Llama_index_sandbox.custom_react_agent.tools.tool_output import CustomToolOutput
from src.Llama_index_sandbox.prompts import (
    QUERY_ENGINE_PROMPT_FORMATTER,
    QUERY_ENGINE_TOOL_ROUTER,
    CONFIRM_FINAL_ANSWER,
    TWITTER_QUERY_ENGINE_PROMPT_FORMATTER,
)
from src.Llama_index_sandbox.utils.utils import timeit

# Prefer core ChatMessage as conversion target
try:
    from llama_index.core.llms import ChatMessage as CoreChatMessage
except ImportError:
    from llama_index.core.base.llms.types import ChatMessage as CoreChatMessage  # type: ignore


def ensure_core_chat_messages(messages):
    """Convert messages to core ChatMessage format."""
    converted = []
    for msg in messages:
        if isinstance(msg, CoreChatMessage):
            converted.append(msg)
        else:
            role = str(getattr(getattr(msg, "role", ""), "value", getattr(msg, "role", "user")))
            converted.append(
                CoreChatMessage(
                    role=role,
                    content=str(getattr(msg, "content", "")),
                    additional_kwargs=getattr(msg, "additional_kwargs", {}),
                )
            )
    return converted


class CustomReActAgent(ReActAgent):
    """Thin wrapper around ReActAgent that avoids Pydantic field errors."""

    def __init__(
        self,
        *,
        tools: list,
        react_chat_formatter,
        llm,
        max_iterations: int = 10,
        memory=None,
        output_parser=None,
        verbose: bool = True,
        callback_manager=None,
    ):
        from llama_index.core.callbacks import CallbackManager as _CB

        # Store internals via private attrs to bypass Pydantic validation
        object.__setattr__(self, "_callback_manager", callback_manager or _CB([]))
        object.__setattr__(self, "_tools", tools or [])
        object.__setattr__(self, "_tools_dict", {
            getattr(t.metadata, "name", f"tool_{i}"): t for i, t in enumerate(tools or [])
        })
        object.__setattr__(self, "_react_chat_formatter", react_chat_formatter)
        object.__setattr__(self, "_output_parser", output_parser)
        object.__setattr__(self, "_llm", llm)
        object.__setattr__(self, "_verbose", verbose)
        object.__setattr__(self, "_max_iterations", max_iterations)

        # Memory – best effort construction
        try:
            from llama_index.core.memory import ChatMemoryBuffer
            mem = memory or ChatMemoryBuffer.from_defaults(chat_history=[], llm=llm)
        except Exception:
            mem = memory
        object.__setattr__(self, "_memory", mem)

        # Optional legacy-style flag used elsewhere
        object.__setattr__(self, "verbose", verbose)

    # --- properties / helpers ---

    @property
    def callback_manager(self):
        """Expose read-only callback manager so @trace_method can access it."""
        return getattr(self, "_callback_manager", None)

    def _get_tools(self, _=None):
        return getattr(self, "_tools", [])

    def reset(self):
        """Clear conversation memory between questions."""
        try:
            self._memory.reset()
        except Exception:
            object.__setattr__(self, "_memory", None)

    # --- main API ---

    @trace_method("chat")
    @timeit
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Union[AgentChatResponse, Tuple[AgentChatResponse, str]]:
        """Chat entrypoint used by your pipeline."""
        if chat_history is not None and getattr(self, "_memory", None) is not None:
            self._memory.set(chat_history)

        # Optionally append router info
        # message_with_tool_description = f"{message}\n{QUERY_ENGINE_TOOL_ROUTER}"
        # self._memory.put(ChatMessage(content=message_with_tool_description, role="user"))
        self._memory.put(ChatMessage(content=message, role="user"))

        current_reasoning: List[BaseReasoningStep] = []
        last_metadata = None
        response = None

        # ReAct loop
        for _ in range(self._max_iterations):
            input_chat = self._react_chat_formatter.format(
                tools=self._get_tools(_),
                chat_history=self._memory.get(),
                current_reasoning=current_reasoning,
            )

            # Debug dump
            logging.info("=" * 50)
            logging.info("DEBUG: Messages being sent to LLM")
            logging.info(f"Count: {len(input_chat)}")
            for i, msg in enumerate(input_chat):
                role_str = str(getattr(getattr(msg, "role", ""), "value", getattr(msg, "role", "")))
                preview = (msg.content or "")[:500 if role_str == "system" else 200]
                logging.info(f"Message {i} [{role_str}]: {preview}...")
            logging.info("=" * 50)

            if (last_metadata is None) or (len(input_chat) == 2):
                if os.environ.get("ENGINEER_CONTEXT_IN_TOOL_RESPONSE") == "True":
                    input_chat[-1].content += f"\n {AVOID_CITING_CONTEXT}"
                logging.info(f"LLM temperature: {getattr(self._llm, 'temperature', 'n/a')}")
                chat_response = self._llm.chat(ensure_core_chat_messages(input_chat))
            else:
                # Manual stitch if you want to skip an extra LLM call
                chat_response = ChatResponse(
                    message=ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=(
                            "Thought: I can answer without using any more tools.\n"
                            f"Answer:{str(input_chat).replace('Observation: ', '')}"
                        ),
                    )
                )

            chat_response_copy = copy.deepcopy(chat_response)

            # Force user question into Action Input
            response_content = chat_response_copy.raw.choices[0].message.content
            import re
            m = re.search(r"Action Input:\s*(\{.*\})", response_content, re.DOTALL)
            if m:
                try:
                    action_input_part = response_content.split("Action Input:")[1].strip()
                    action_input_json = json.loads(action_input_part)

                    current_date = datetime.now().strftime("%Y-%m-%d")
                    is_twitter = os.environ.get("TWITTER_BOT", "FALSE") == "TRUE"
                    prompt_tmpl = (
                        TWITTER_QUERY_ENGINE_PROMPT_FORMATTER
                        if is_twitter else QUERY_ENGINE_PROMPT_FORMATTER
                    )
                    augmented_message = prompt_tmpl.format(
                        current_date=current_date,
                        user_raw_input=message,
                        llm_reasoning_on_user_input=action_input_json.get("input", ""),
                    )
                    action_input_json["input"] = augmented_message

                    response_content = response_content.replace(
                        action_input_part, json.dumps(action_input_json)
                    )
                    chat_response_copy.raw.choices[0].message.content = response_content
                    chat_response_copy.message.content = response_content
                except Exception as e:
                    logging.error(f"Failed to modify Action Input: {e}")

            # Process tool actions
            if os.environ.get("ENVIRONMENT") == "LOCAL":
                logging.info(f"Starting _process_actions with chat_response_copy: {chat_response_copy}")

            if last_metadata is not None:
                reasoning_steps, is_done, last_metadata = self._process_actions(
                    output=chat_response_copy, last_metadata=last_metadata
                )
            else:
                reasoning_steps, is_done, last_metadata = self._process_actions(
                    output=chat_response_copy
                )
            current_reasoning.extend(reasoning_steps)

            if is_done or last_metadata is not None:
                if isinstance(current_reasoning[-1], ResponseReasoningStep):
                    response = AgentChatResponse(response=current_reasoning[-1].response)
                elif isinstance(current_reasoning[-1], ObservationReasoningStep):
                    response = AgentChatResponse(response=current_reasoning[-1].observation)
                else:
                    logging.error(
                        "Last reasoning step is neither ObservationReasoningStep nor ResponseReasoningStep."
                    )
                    raise RuntimeError("Unexpected reasoning step type.")
                break

        if not response:
            response = self._get_response(current_reasoning)

        self._memory.put(ChatMessage(content=response.response, role=MessageRole.ASSISTANT))
        return response, last_metadata

    # --- fallbacks for core/legacy differences --------------------------------

    def _extract_reasoning_step(self, output) -> Tuple[Optional[str], List[BaseReasoningStep], bool]:
        """
        Minimal ReAct parser:
        - Extract optional 'Thought:'.
        - If it finds `Final Answer:` or `Answer:`, returns ResponseReasoningStep(thought=..., response=...) and is_done=True.
        - Else if it finds `Action:` and `Action Input:`, returns ActionReasoningStep(thought=..., action=..., action_input=...).
        """
        # Pull content from ChatResponse (supports core/legacy)
        content = ""
        try:
            content = output.message.content
        except Exception:
            try:
                # core-style OpenAI adapter often exposes .raw.choices[0].message.content
                content = output.raw.choices[0].message.content  # type: ignore[attr-defined]
            except Exception:
                try:
                    content = output.raw["choices"][0]["message"]["content"]  # dictionary-style
                except Exception:
                    pass

        steps: List[BaseReasoningStep] = []
        is_done = False

        import re, json

        # Parse an optional Thought line (take the last occurrence above Action/Answer)
        thought = None
        for m in re.finditer(r"Thought\s*:\s*(.*)", content, re.IGNORECASE):
            thought = m.group(1).strip()
        if not thought:
            thought = "Reason about the question and decide whether to call a tool."

        # 1) Final answer?
        m_final = re.search(r"(?:Final Answer|Answer)\s*:\s*(.*)", content, re.DOTALL | re.IGNORECASE)
        if m_final:
            answer = m_final.group(1).strip()
            steps.append(ResponseReasoningStep(thought=thought, response=answer))
            return None, steps, True

        # 2) Action + Action Input?
        m_action = re.search(r"Action\s*:\s*([A-Za-z0-9_\-\.]+)", content)
        m_input = re.search(r"Action Input\s*:\s*(\{.*\})", content, re.DOTALL)
        if m_action and m_input:
            action_name = m_action.group(1).strip()
            raw_ai = m_input.group(1).strip()
            try:
                action_input = json.loads(raw_ai)
            except Exception:
                # Not valid JSON? Wrap raw as text so tool can still get something
                action_input = {"input": raw_ai}
            steps.append(ActionReasoningStep(thought=thought, action=action_name, action_input=action_input))
            return None, steps, False

        # 3) Nothing recognized – no step this turn
        return None, steps, False

    def _get_response(self, steps: List[BaseReasoningStep]) -> AgentChatResponse:
        """
        Fallback finalizer if your base agent doesn’t provide one.
        Prefer last ResponseReasoningStep, else last Observation, else empty.
        """
        for step in reversed(steps):
            try:
                if hasattr(step, "response") and step.response:
                    # step may already be ResponseReasoningStep; just return its text
                    return AgentChatResponse(response=str(step.response))
                if hasattr(step, "observation") and step.observation:
                    return AgentChatResponse(response=str(step.observation))
            except Exception:
                continue
        # As a last resort, provide something well-formed
        return AgentChatResponse(response="")

    @timeit
    def confirm_response(self, question: str, response: str, sources: str) -> AgentChatResponse:
        self._llm.model = "gpt-4-0613"
        if not sources:
            return AgentChatResponse(response=response, sources=[])
        final_input = ChatMessage(
            role="user",
            content=CONFIRM_FINAL_ANSWER.format(question=question, response=response, sources=sources),
        )
        chat_response = self._llm.chat([final_input])
        final_answer = chat_response.raw.choices[0]["message"]["content"]
        return AgentChatResponse(response=final_answer, sources=[])

    @timeit
    def _process_actions(
        self, output: ChatResponse, last_metadata: Optional[str] = None
    ) -> Tuple[List[BaseReasoningStep], bool, str]:
        _, current_reasoning, is_done = self._extract_reasoning_step(output)
        if is_done:
            return current_reasoning, True, last_metadata

        reasoning_step = cast(ActionReasoningStep, current_reasoning[-1])
        self._tools_dict = {tool.metadata.name: tool for tool in self._get_tools(_)}
        tool = self._tools_dict[reasoning_step.action]

        with self.callback_manager.event(
            CBEventType.FUNCTION_CALL,
            payload={EventPayload.FUNCTION_CALL: reasoning_step.action_input, EventPayload.TOOL: tool.metadata},
        ) as event:
            tool_output = tool.call(**reasoning_step.action_input)

            if isinstance(tool_output, CustomToolOutput):
                formatted_metadata = tool_output.get_formatted_metadata()
            else:
                formatted_metadata = "Metadata not available."

            event.on_end(
                payload={
                    EventPayload.FUNCTION_OUTPUT: str(tool_output),
                    ExtendedEventPayload.FORMATTED_METADATA: formatted_metadata,
                }
            )

        observation_content = str(tool_output)
        observation_step = ObservationReasoningStep(observation=observation_content)
        last_metadata = (
            tool_output.get_formatted_metadata()
            if isinstance(tool_output, CustomToolOutput)
            else formatted_metadata
        )
        current_reasoning.append(observation_step)

        if self._verbose and os.environ.get("ENVIRONMENT") == "LOCAL":
            print_text(f"{observation_step.get_content()}\n", color="blue")
            print_text(f"{last_metadata}\n", color="blue")
            logging.info(f"{observation_step.get_content()}")
            logging.info(f"{last_metadata}")

        return current_reasoning, False, last_metadata
