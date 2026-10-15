#!/usr/bin/env python3
"""Extract caller turns from plain-text call transcripts.

Expected format includes blocks such as:
  0:00 · Call connected
  0:07Caller
  Hello.
  0:15Agent Deepanshu
  ...

Only lines explicitly headed by a Caller marker are extracted. Agent speech and
call events are excluded. Review the output before training: ASR text is not truth.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CALL_START = re.compile(r"^\s*\d+:\d+\s*·\s*Call connected\s*$", re.IGNORECASE)
SPEAKER = re.compile(r"^\s*\d+:\d+\s*(Caller|Agent(?:\s+.*)?)\s*$", re.IGNORECASE)
EVENT = re.compile(r"^\s*\d+:\d+\s*·\s*(?:Caller hung up|Call ended|.*)$", re.IGNORECASE)


def extract_caller_turns(raw_text: str) -> list[dict]:
    rows: list[dict] = []
    call_id = 0
    speaker = None
    utterance_lines: list[str] = []
    start_line = None

    def flush():
        nonlocal utterance_lines, start_line
        if speaker == "caller" and utterance_lines:
            text = " ".join(part.strip() for part in utterance_lines if part.strip()).strip()
            if text:
                rows.append({
                    "call_id": call_id,
                    "source_line": start_line,
                    "text": text,
                })
        utterance_lines = []
        start_line = None

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if CALL_START.match(line):
            flush()
            call_id += 1
            speaker = None
            continue
        match = SPEAKER.match(line)
        if match:
            flush()
            speaker = "caller" if match.group(1).lower() == "caller" else "agent"
            start_line = line_no + 1
            continue
        if EVENT.match(line):
            flush()
            speaker = None
            continue
        if not line.strip():
            # Keep multi-line utterances together; flush on the next speaker/event.
            continue
        if speaker in {"caller", "agent"}:
            utterance_lines.append(line.strip())
    flush()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source transcript .txt")
    parser.add_argument("--output", required=True, help="Output CSV path; keep private if source data is private")
    args = parser.parse_args()
    raw = Path(args.input).read_text(encoding="utf-8", errors="replace")
    rows = extract_caller_turns(raw)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["call_id", "source_line", "text"]).to_csv(out, index=False)
    print(f"Extracted {len(rows)} caller turns across {len({r['call_id'] for r in rows})} detected calls.")
    print(f"Saved to {out}. Inspect for transcript-format exceptions before using it.")


if __name__ == "__main__":
    main()
