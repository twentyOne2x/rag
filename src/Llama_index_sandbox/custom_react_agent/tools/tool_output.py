import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

# ✅ core ToolOutput instead of legacy
from llama_index.core.tools import ToolOutput
from pydantic import Field

from src.rag_v2.logging_utils import format_metadata

def _hms_to_seconds(hms: str | None) -> int:
    if not hms:
        return -1
    try:
        parts = [int(x) for x in hms.split(":")]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            return -1
        return max(0, h * 3600 + m * 60 + s)
    except Exception:
        return -1


def _add_time_param(url: str, seconds: int) -> str:
    """Append/replace t=Ns on the URL."""
    try:
        u = urlparse(url)
        q = parse_qs(u.query)
        q["t"] = [f"{max(0, int(seconds))}s"]
        new_q = urlencode(q, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))
    except Exception:
        sep = "&" if "?" in (url or "") else "?"
        return f"{url}{sep}t={max(0, int(seconds))}s"


def _format_timestamp_range(start_hms: str | None, end_hms: str | None) -> str:
    if start_hms and end_hms:
        return f"{start_hms}–{end_hms}"
    return (start_hms or "") or ""


class CustomToolOutput(ToolOutput):
    all_formatted_metadata: str = Field(default="")

    def __init__(self, **data):
        super().__init__(**data)
        try:
            self.all_formatted_metadata = format_metadata(self.raw_output)
        except Exception:
            self.all_formatted_metadata = "No metadata available"
            print("Warning: raw_output may not contain metadata as expected.")

    def __str__(self) -> str:
        return f"{self.content}"

    def get_formatted_metadata(self) -> str:
        return self.all_formatted_metadata


def log_and_store(store_response_fn, query_str, response, chatbot: bool):
    all_formatted_metadata = format_metadata(response)
    if chatbot:
        msg = (
            f"The answer to the question {query_str} is: \n{response}\n\nFetched based on the following sources: \n"
            f"{all_formatted_metadata}\n"
        )
    else:
        msg = (
            f"The answer to [{query_str}] is: \n\n```\n{response}\n\n\nFetched based on the following sources/content: \n"
            f"{all_formatted_metadata}\n```"
        )
        logging.info(f"[Shown to client] {msg}")
    return msg, all_formatted_metadata
