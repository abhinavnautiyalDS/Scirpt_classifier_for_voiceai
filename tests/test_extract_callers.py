from src.extract_callers import extract_caller_turns


def test_extracts_caller_and_excludes_agent_and_call_events():
    transcript = """Call flow
0:00 · Call connected
0:00Agent Deepanshu
Good morning BNI
0:07Caller
Hello.
0:15Agent Deepanshu
नमस्ते, कैसे हैं?
0:20Caller
I need more details.
0:22 · Caller hung up
"""
    rows = extract_caller_turns(transcript)
    assert len(rows) == 2
    assert rows[0]["call_id"] == 1
    assert rows[0]["text"] == "Hello."
    assert rows[1]["text"] == "I need more details."
    assert all("नमस्ते" not in row["text"] for row in rows)
