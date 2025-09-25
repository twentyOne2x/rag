import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ✅ Use ONLY legacy callback interfaces/types
from llama_index.legacy.callbacks.base import BaseCallbackHandler
from llama_index.legacy.callbacks.schema import CBEventType, EventPayload
from llama_index.legacy.llms import MessageRole

from src.Llama_index_sandbox import root_dir
from src.Llama_index_sandbox.custom_react_agent.callbacks.schema import ExtendedEventPayload
from src.Llama_index_sandbox.custom_react_agent.tools.query_engine_prompts import TEXT_QA_SYSTEM_PROMPT
from src.Llama_index_sandbox.prompts import QUERY_ENGINE_TOOL_ROUTER
from src.Llama_index_sandbox.utils.utils import get_last_index_embedding_params, timeit
from src.Llama_index_sandbox import globals as glb

# Snapshot of current index/embedding params (used only for logging metadata)
embedding_model_name, text_splitter_chunk_size, chunk_overlap, _ = get_last_index_embedding_params()


# --------------------------- Helpers -----------------------------------------

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _first_system_text(messages: List[Any]) -> Optional[str]:
    if not messages:
        return None
    msg0 = messages[0]
    try:
        return msg0.content if getattr(msg0, "role", None) == MessageRole.SYSTEM else None
    except Exception:
        return None


def _last_user_text(messages: List[Any]) -> Optional[str]:
    if not messages:
        return None
    try:
        for m in reversed(messages):
            if getattr(m, "role", None) == MessageRole.USER:
                return m.content
    except Exception:
        pass
    return None


def _ascii_preview(value: Any, max_len: int = 300) -> str:
    """
    Compact, ASCII-safe preview for console logs.
    Truncates and replaces non-ASCII characters so latin-1 consoles won't crash.
    """
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s.encode("ascii", "replace").decode("ascii")


def _summarize_entry_for_console(event_label: str, entry: Dict[str, Any]) -> str:
    """
    Produce a short, ASCII-safe one-liner for console logging.
    Shows keys and tiny previews of the noisiest fields.
    """
    try:
        keys = ", ".join(sorted(entry.keys()))
    except Exception:
        keys = "(unavailable)"
    previews: List[str] = []
    for k in ("tool_output", "instructions", "retrieved_chunk", "LLM_response", "LLM_input", "user_raw_input"):
        if k in entry and entry[k] is not None:
            previews.append(f"{k}={_ascii_preview(entry[k])}")
    preview_str = " | ".join(previews[:3])  # keep it short
    return f"{event_label}: keys=[{keys}]{(' | ' + preview_str) if preview_str else ''}"


# --------------------- JSONLoggingHandler ------------------------------------

class JSONLoggingHandler(BaseCallbackHandler):
    """Legacy callback handler that emits structured JSON logs."""

    def __init__(
        self,
        event_starts_to_ignore: List[CBEventType],
        event_ends_to_ignore: List[CBEventType],
        log_name: str,
        similarity_top_k: int,
    ) -> None:
        # ✅ FIX 1: pass required args to base class
        super().__init__(event_starts_to_ignore, event_ends_to_ignore)

        self.current_section: Optional[List[Dict[str, Any]]] = None
        self.current_logs: List[Dict[str, Any]] = []

        _safe_mkdir(f"{root_dir}/logs/json")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = f"{root_dir}/logs/json/{ts}_{log_name}_{similarity_top_k}.json"

        # Always write JSON as UTF-8, preserving Unicode characters
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.current_logs, f, ensure_ascii=False)

    # ---- Internal helpers -------------------------------------------------

    def _rewrite_log_file(self) -> None:
        # ✅ ensure UTF-8 when rewriting the log file and keep Unicode as-is
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.current_logs, f, indent=4, ensure_ascii=False)

    def _append_entry(self, entry: Dict[str, Any]) -> None:
        if not entry:
            return
        if self.current_section is not None:
            self.current_section.append(entry)
        else:
            self.current_logs.append(entry)
        self._rewrite_log_file()

    @timeit
    def _parse_message_content(self, message_content: str) -> Tuple[Optional[str], Optional[str]]:
        context_start_delim = "New Context:"
        context_end_delim = "Query:"
        answer_start_delim = "Original Answer:"
        answer_end_delim = "New Answer:"

        if "Observation:" in message_content:
            return message_content, None

        try:
            context_start = message_content.index(context_start_delim) + len(context_start_delim)
            context_end = message_content.index(context_end_delim)
            answer_start = message_content.index(answer_start_delim) + len(answer_start_delim)
            answer_end = message_content.index(answer_end_delim)

            retrieved_context = message_content[context_start:context_end].strip()
            previous_answer = message_content[answer_start:answer_end].strip()
            return retrieved_context, previous_answer
        except ValueError as e:
            logging.warning(f"_parse_message_content: could not parse message content: {e}")
            return None, None

    # ---- BaseCallbackHandler interface ------------------------------------

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        # ✅ FIX 2: ensure we return an id
        eid = event_id or str(uuid.uuid4())

        if event_type in self.event_starts_to_ignore:
            return eid

        entry: Dict[str, Any] = {}
        payload = payload or {}

        if event_type == CBEventType.LLM:
            messages: List[Any] = payload.get(EventPayload.MESSAGES, []) or []
            serialized = payload.get(EventPayload.SERIALIZED, {}) or {}

            last_user = _last_user_text(messages)
            first_system = _first_system_text(messages)

            if last_user and QUERY_ENGINE_TOOL_ROUTER in last_user:
                user_raw_input = last_user.replace(f"\n{QUERY_ENGINE_TOOL_ROUTER}", "")
                entry = {
                    "event_type": f"{event_type.name} start",
                    "model_params": serialized,
                    "embedding_model_parameters": {
                        "embedding_model_name": embedding_model_name,
                        "text_splitter_chunk_size": text_splitter_chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "number_of_chunks_to_retrieve": getattr(glb, "NUMBER_OF_CHUNKS_TO_RETRIEVE", None),
                    },
                    "user_raw_input": user_raw_input,
                    "LLM_input": last_user,
                }

            elif last_user and "Context information is below." in last_user:
                has_expected_text = False
                if first_system and TEXT_QA_SYSTEM_PROMPT.content in first_system:
                    has_expected_text = True
                elif TEXT_QA_SYSTEM_PROMPT.content in last_user:
                    has_expected_text = True
                if not has_expected_text:
                    logging.debug(
                        "Q&A system prompt not found in SYSTEM role; assuming it is inlined in USER message."
                    )
                entry = {"event_type": f"{event_type.name} start", "tool_output": last_user}

            elif last_user:
                retrieved_context, previous_answer = self._parse_message_content(last_user)
                entry = {
                    "event_type": f"{event_type.name} start",
                    "retrieved_context": retrieved_context,
                    "previous_answer": previous_answer,
                }
            else:
                logging.info(f"WARNING: on_event_start: {event_type.name} had no USER message; skipping details.")

        elif event_type == CBEventType.FUNCTION_CALL:
            entry = {"event_type": f"{event_type.name} start", "function_call": []}
            # when starting a function call section, open a new sublist
            self.current_logs.append(entry)
            self._rewrite_log_file()
            self.current_section = entry["function_call"]
            # compact, ASCII-safe console line
            logging.info(_summarize_entry_for_console("on_event_start", entry))
            return eid  # already appended & section set

        elif event_type == CBEventType.TEMPLATING:
            template_vars = payload.get(EventPayload.TEMPLATE_VARS, {}) or {}
            template = payload.get(EventPayload.TEMPLATE, "") or ""
            entry = {
                "event_type": f"{event_type.name} start",
                "instructions": template,
                "retrieved_chunk": template_vars,
            }

        else:
            logging.info(f"WARNING: on_event_start: unhandled event_type {event_type.name}")

        # persist full entry to JSON
        self._append_entry(entry)
        # print compact preview to console
        logging.info(_summarize_entry_for_console("on_event_start", entry))
        return eid

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        if event_type in self.event_ends_to_ignore:
            return

        payload = payload or {}
        entry: Dict[str, Any] = {}

        if event_type == CBEventType.LLM:
            response = payload.get(EventPayload.RESPONSE, None)
            try:
                msg = getattr(response, "message", None)
                role = getattr(msg, "role", None)
                content = getattr(msg, "content", "")
            except Exception:
                role, content = None, ""

            if role == MessageRole.ASSISTANT:
                if content.startswith("Thought: I need to use a tool to help me answer the question."):
                    entry = {"event_type": f"{event_type.name} end", "LLM_response": content}
                else:
                    entry = {
                        "event_type": f"{event_type.name} end",
                        "LLM_response": content,
                        "subjective_grade_1_to_10": "",
                    }
            else:
                logging.info(
                    f"WARNING: on_event_end: {event_type.name} assistant message not found; skipping detailed log."
                )

        elif event_type == CBEventType.FUNCTION_CALL:
            entry = {
                "event_type": f"{event_type.name} end",
                "tool_output": payload.get(EventPayload.FUNCTION_OUTPUT),
                "metadata": payload.get(ExtendedEventPayload.FORMATTED_METADATA),
            }
            self.current_section = None

        elif event_type == CBEventType.TEMPLATING:
            entry = {"event_type": f"{event_type.name} end"}

        else:
            logging.info(f"WARNING: on_event_end: unhandled event_type {event_type.name}")

        # persist full entry to JSON
        self._append_entry(entry)
        # print compact preview to console
        logging.info(_summarize_entry_for_console("on_event_end", entry))

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(self, trace_id: Optional[str] = None, trace_map: Optional[Dict[str, List[str]]] = None) -> None:
        return
