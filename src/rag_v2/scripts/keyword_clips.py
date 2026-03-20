from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag_v2.vector_store.keyword_clips import scan_keyword_clips_qdrant


def _build_channel_filter(args: argparse.Namespace) -> Optional[Dict[str, List[str]]]:
    def _clean_list(xs):
        out = []
        for x in xs or []:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    payload = {
        "include_ids": _clean_list(args.include_channel_id),
        "exclude_ids": _clean_list(args.exclude_channel_id),
        "include_names": _clean_list(args.include_channel_name),
        "exclude_names": _clean_list(args.exclude_channel_name),
    }
    payload = {k: v for k, v in payload.items() if v}
    return payload or None


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump all transcript clips that match a keyword/phrase (Qdrant only).")
    ap.add_argument("--query", required=True, help="Keyword or phrase (e.g., credit, 'onchain credit').")
    ap.add_argument("--phrase", action="append", default=None, help="Additional phrase to match (repeatable).")
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "videos"))
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (scan entire collection).")
    ap.add_argument("--offset", default=None, help="Qdrant scroll offset cursor (advanced).")
    ap.add_argument("--out", default=None, help="Write full results as JSONL to this path (recommended).")

    ap.add_argument("--include-channel-id", action="append", default=None)
    ap.add_argument("--exclude-channel-id", action="append", default=None)
    ap.add_argument("--include-channel-name", action="append", default=None)
    ap.add_argument("--exclude-channel-name", action="append", default=None)

    args = ap.parse_args()

    channel_filter = _build_channel_filter(args)
    res = scan_keyword_clips_qdrant(
        query=str(args.query),
        phrases=args.phrase,
        namespace=str(args.namespace),
        limit=int(args.limit),
        offset=str(args.offset) if args.offset else None,
        channel_filter=channel_filter,
    )

    rows = list(res.get("results") or [])

    # Stats
    by_parent = Counter(str(r.get("parent_id") or "") for r in rows if r.get("parent_id"))
    by_speaker = Counter(
        str((r.get("speaker_identity") or r.get("speaker_name") or r.get("speaker") or "")).strip()
        for r in rows
        if (r.get("speaker_identity") or r.get("speaker_name") or r.get("speaker"))
    )

    print(
        json.dumps(
            {
                "ok": bool(res.get("ok")),
                "namespace": res.get("namespace"),
                "query": res.get("query"),
                "match_phrases_norm": res.get("match_phrases_norm"),
                "match_tokens_norm": res.get("match_tokens_norm"),
                "scanned": res.get("scanned"),
                "matched": res.get("matched"),
                "unique_parents": res.get("unique_parents"),
                "unique_speakers": res.get("unique_speakers"),
                "exhausted": res.get("exhausted"),
                "next_offset": res.get("next_offset"),
            },
            indent=2,
            sort_keys=True,
        )
    )

    # Top speakers summary (small, human-readable).
    if by_speaker:
        top = by_speaker.most_common(25)
        print("\nTop speakers by matching clips:")
        for name, cnt in top:
            if not name:
                continue
            print(f"- {name}: {cnt}")

    # Output
    if args.out:
        out_path = Path(str(args.out)).expanduser()
        _write_jsonl(out_path, rows)
        print(f"\nwrote_jsonl={out_path}")
    else:
        # Show a small sample so the command is useful without writing files.
        print("\nSample clips:")
        for r in rows[:10]:
            title = r.get("parent_title") or r.get("title") or r.get("parent_id")
            hms = " - ".join([x for x in [r.get("start_hms"), r.get("end_hms")] if x])
            spk = r.get("speaker_identity") or r.get("speaker_name") or r.get("speaker") or ""
            url = r.get("clip_url") or r.get("url") or ""
            print(f"- {title} [{spk}] ({hms}) {url}")


if __name__ == "__main__":
    main()
