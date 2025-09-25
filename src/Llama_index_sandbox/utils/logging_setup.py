# src/Llama_index_sandbox/utils/logging_setup.py
from __future__ import annotations

import io
import logging
import os
import sys
from typing import Optional


# ---------- ANSI color helpers ----------

RESET = "\x1b[0m"
BOLD = "\x1b[1m"

# Basic palette (safe in most terminals & PyCharm)
FG = {
    "black":   "\x1b[30m",
    "red":     "\x1b[31m",
    "green":   "\x1b[32m",
    "yellow":  "\x1b[33m",
    "blue":    "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan":    "\x1b[36m",
    "white":   "\x1b[37m",
}

LEVEL_STYLE = {
    logging.DEBUG:   FG["red"],
    logging.INFO:    FG["red"],
    logging.WARNING: FG["red"],
    logging.ERROR:   FG["red"],
    logging.CRITICAL: BOLD + FG["red"],
}


def _supports_ansi(stream) -> bool:
    # PyCharm usually supports ANSI; also detect TTY.
    try:
        if hasattr(stream, "isatty") and stream.isatty():
            return True
    except Exception:
        pass
    # Heuristic: PyCharm sets this env
    if os.environ.get("PYCHARM_HOSTED"):
        return True
    # VSCode integrated terminal, etc.
    if os.environ.get("TERM") in {"xterm-color", "xterm-256color"}:
        return True
    return False


# ---------- Filters / Handlers ----------

class AsciiSafeFilter(logging.Filter):
    """
    Forces record.msg/args to be ASCII-safe so latin-1 consoles never crash.
    Applies to *all* handlers when attached to root logger.
    """

    @staticmethod
    def _safe(obj):
        try:
            if isinstance(obj, str):
                # Keep info but escape non-ascii: "☀️" -> "\u2600\uFE0F"
                return obj.encode("ascii", "backslashreplace").decode("ascii")
            return obj
        except Exception:
            return repr(obj)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._safe(record.msg)
        if record.args:
            try:
                if isinstance(record.args, dict):
                    record.args = {k: self._safe(v) for k, v in record.args.items()}
                else:
                    record.args = tuple(self._safe(a) for a in record.args)
            except Exception:
                record.args = ()
        return True


class SafeStreamHandler(logging.StreamHandler):
    """
    Stream handler that never raises UnicodeEncodeError.
    If the stream can't encode a character, it falls back to backslash escaping.
    """

    def __init__(self, stream: Optional[io.TextIOBase] = None) -> None:
        super().__init__(stream or sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            try:
                self.stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                enc = getattr(self.stream, "encoding", None) or "utf-8"
                if hasattr(self.stream, "buffer"):
                    self.stream.buffer.write((msg + self.terminator).encode(enc, "backslashreplace"))
                else:
                    safe = (msg + self.terminator).encode(enc, "backslashreplace").decode(enc, "replace")
                    self.stream.write(safe)
            self.flush()
        except Exception:
            self.handleError(record)


class ColorFormatter(logging.Formatter):
    """
    Color the entire rendered log line by level.
    Falls back to plain text if color is disabled or unsupported.
    """

    def __init__(self, fmt: str, datefmt: str, enable_color: bool, stream_supports_ansi: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.enable_color = bool(enable_color and stream_supports_ansi)

    def format(self, record: logging.LogRecord) -> str:
        # Render once using the base formatter
        rendered = super().format(record)

        if not self.enable_color:
            return rendered

        style = LEVEL_STYLE.get(record.levelno, "")
        if not style:
            return rendered

        # Colorize the entire line (prefix + message), then reset
        return f"{style}{rendered}{RESET}"


# ---------- Internals ----------

_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _mkdir_p(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _looks_like_path(s: str) -> bool:
    return os.path.sep in s or s.endswith(".log")


# ---------- Public API ----------

def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Optional[str] = None,
    console: bool = True,
    color: bool = True,  # <-- NEW
    fmt: str = _FORMAT,
    datefmt: str = _DATEFMT,
) -> None:
    """
    Configure the root logger (idempotent).

    Usage:
        configure_logging("/path/to/run.log")               # path-only call
        configure_logging(logging.DEBUG, log_file=path)     # explicit level + file
        configure_logging("DEBUG", log_file=path)           # level as string

    - Adds an ASCII-safe filter at the root so *all* handlers are protected.
    - Adds a UTF-8 file handler if log_file is provided.
    - Adds a SafeStreamHandler console handler if console=True.
    - Console output can be colored with ANSI (color=True, default).

    Subsequent calls:
      - Won't clear existing handlers.
      - Will add a file handler if a *new* log_file path is provided.
    """

    # Prefer UTF-8 if possible
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    # Guard against accidental positional (path-as-level) use
    if isinstance(level, str):
        if _looks_like_path(level):
            log_file = level
            level = logging.INFO
        else:
            level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()

    # First-time configuration?
    already_configured = getattr(root, "_ascii_safe_configured", False)

    if not already_configured:
        root.setLevel(level)
        root.addFilter(AsciiSafeFilter())

        # Console handler
        if console:
            ch = SafeStreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(
                ColorFormatter(
                    fmt=fmt,
                    datefmt=datefmt,
                    enable_color=color,
                    stream_supports_ansi=_supports_ansi(ch.stream),
                )
            )
            ch._is_app_console = True  # marker
            root.addHandler(ch)

        # Optional file handler
        if log_file:
            _mkdir_p(os.path.dirname(log_file))
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            fh._is_app_run_file = True  # marker
            fh._app_path = os.path.abspath(log_file)
            root.addHandler(fh)

        root._ascii_safe_configured = True
        return

    # If already configured:
    try:
        if isinstance(level, int) and level < root.level:
            root.setLevel(level)
    except Exception:
        pass

    # Ensure console exists if requested
    if console and not any(getattr(h, "_is_app_console", False) for h in root.handlers):
        ch = SafeStreamHandler(sys.stdout)
        ch.setLevel(level if isinstance(level, int) else logging.INFO)
        ch.setFormatter(
            ColorFormatter(
                fmt=fmt,
                datefmt=datefmt,
                enable_color=color,
                stream_supports_ansi=_supports_ansi(ch.stream),
            )
        )
        ch._is_app_console = True
        root.addHandler(ch)

    # Add file handler if a *new* path is provided
    if log_file:
        abs_path = os.path.abspath(log_file)
        have_this_file = any(
            getattr(h, "_is_app_run_file", False) and getattr(h, "_app_path", "") == abs_path
            for h in root.handlers
        )
        if not have_this_file:
            _mkdir_p(os.path.dirname(log_file))
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level if isinstance(level, int) else logging.INFO)
            fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
            fh._is_app_run_file = True
            fh._app_path = abs_path
            root.addHandler(fh)


def add_section_file_logger(log_file_path: str) -> logging.Handler:
    """
    Add a *section-specific* UTF-8 file handler (e.g., create_index_..., ask_questions_...).
    Returns the handler so the caller can remove it later if desired.

    This does not modify or clear existing handlers/filters.
    """
    _mkdir_p(os.path.dirname(log_file_path))
    h = logging.FileHandler(log_file_path, encoding="utf-8")
    h.setLevel(logging.getLogger().level)
    h.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
    h._is_app_section = True
    h._app_path = os.path.abspath(log_file_path)
    logging.getLogger().addHandler(h)
    return h
