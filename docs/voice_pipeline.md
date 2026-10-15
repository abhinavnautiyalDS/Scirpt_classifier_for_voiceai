# Voice-agent integration

## Recommended flow

1. Receive the caller's completed/endpointed ASR utterance from the STT service.
2. Normalize whitespace only. Preserve the original transcript for debugging; do not translate before classification.
3. Run the language classifier and obtain one label: HINGLISH, ENGLISH, NEUTRAL, or UNCERTAIN.
4. Pass the label and utterance to `LanguageSwitchController`.
5. Change the agent's response-language setting only when the controller returns `switched=true`.
6. Keep neutral/uncertain utterances in the current language. Log labels, decisions, and timing without logging sensitive transcript text unless required and authorized.

## Minimal example

```python
from src.policy import LanguageSwitchController

controller = LanguageSwitchController(initial_language="HINGLISH", confirmations=2)

# Replace these example labels with actual classifier predictions.
for text, label in [
    ("Please send the details tomorrow", "ENGLISH"),
    ("Could you share the available plans", "ENGLISH"),
]:
    decision = controller.update(text, label)
    print(decision)
    if decision["switched"]:
        # Update the voice-agent response-language setting here.
        pass
```

## Integration notes

- The classifier output is not permission to switch. Neutral phrase overrides and a minimum evidence length are applied separately.
- Two consecutive candidate utterances are required by default. Tune this with call-level tests; do not remove the stability gate merely to improve perceived responsiveness.
- The language classifier only sees text. It cannot measure STT word error rate, audio-language detection accuracy, interruption timing, or end-to-end voice latency.
- For a production integration, benchmark time from final ASR transcript to decision and include model loading separately from per-utterance latency.
- Keep a human-reviewed, call-grouped test set. Report per-class metrics and neutral false-switch rate in addition to accuracy.
