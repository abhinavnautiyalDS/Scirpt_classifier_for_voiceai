"""Deterministic language-switch policy, kept separate from model classification."""
from __future__ import annotations

import re
from dataclasses import dataclass

LABELS = frozenset({"HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"})
NEUTRAL_PHRASES = frozenset({
    "hi", "hello", "hey", "bye", "goodbye", "good bye", "good morning", "good evening",
    "okay", "ok", "yes", "yeah", "yep", "no", "nope", "right", "sure", "hmm", "mm",
    "thank you", "thanks", "ठीक है", "हाँ", "हां", "नहीं", "अच्छा", "जी", "हेलो", "नमस्ते",
})


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", normalize_text(text), flags=re.UNICODE))


@dataclass(frozen=True)
class PolicyResult:
    label: str
    switch_candidate: bool
    reason: str


def apply_switch_policy(text: str, predicted_label: str, min_words: int = 3) -> PolicyResult:
    """Decide if a prediction may be considered by a separate stability controller."""
    clean = normalize_text(text)
    label = str(predicted_label or "").strip().upper()
    if label not in LABELS:
        return PolicyResult("UNCERTAIN", False, "invalid model label")
    if not clean:
        return PolicyResult("UNCERTAIN", False, "empty utterance")
    canonical = re.sub(r"[^\w\s]", "", clean.casefold()).strip()
    if canonical in NEUTRAL_PHRASES:
        return PolicyResult("NEUTRAL", False, "generic neutral phrase override")
    if label in {"NEUTRAL", "UNCERTAIN"}:
        return PolicyResult(label, False, "neutral/uncertain predictions never trigger switching")
    if word_count(clean) < min_words:
        return PolicyResult(label, False, f"fewer than {min_words} words")
    return PolicyResult(label, True, "candidate only; require repeated evidence before switching")


class LanguageSwitchController:
    """Require consecutive matching candidate labels before changing language."""

    def __init__(self, initial_language: str = "HINGLISH", confirmations: int = 2, min_words: int = 3):
        initial = initial_language.strip().upper()
        if initial not in {"HINGLISH", "ENGLISH"}:
            raise ValueError("initial_language must be HINGLISH or ENGLISH")
        if confirmations < 1:
            raise ValueError("confirmations must be >= 1")
        self.current_language = initial
        self.confirmations = confirmations
        self.min_words = min_words
        self._candidate = None
        self._streak = 0

    def update(self, text: str, predicted_label: str) -> dict:
        result = apply_switch_policy(text, predicted_label, self.min_words)
        if not result.switch_candidate:
            self._candidate, self._streak = None, 0
            return {
                "current_language": self.current_language,
                "switched": False,
                "candidate_label": result.label,
                "reason": result.reason,
                "streak": 0,
            }
        if result.label == self.current_language:
            self._candidate, self._streak = None, 0
            return {
                "current_language": self.current_language,
                "switched": False,
                "candidate_label": result.label,
                "reason": "prediction matches current language",
                "streak": 0,
            }
        if result.label == self._candidate:
            self._streak += 1
        else:
            self._candidate, self._streak = result.label, 1
        if self._streak >= self.confirmations:
            self.current_language = result.label
            self._candidate, self._streak = None, 0
            return {
                "current_language": self.current_language,
                "switched": True,
                "candidate_label": result.label,
                "reason": "consecutive candidate threshold met",
                "streak": 0,
            }
        return {
            "current_language": self.current_language,
            "switched": False,
            "candidate_label": result.label,
            "reason": "waiting for consecutive confirmation",
            "streak": self._streak,
        }
