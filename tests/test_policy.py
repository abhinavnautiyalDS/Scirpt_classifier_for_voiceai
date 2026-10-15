from src.policy import LanguageSwitchController, apply_switch_policy


def test_neutral_phrase_never_switches():
    result = apply_switch_policy("Hello!", "ENGLISH")
    assert result.label == "NEUTRAL"
    assert result.switch_candidate is False


def test_uncertain_never_switches():
    result = apply_switch_policy("ನನಗೆ ಸಹಾಯ ಬೇಕು", "UNCERTAIN")
    assert result.switch_candidate is False


def test_short_utterance_does_not_switch():
    result = apply_switch_policy("Yes sir", "ENGLISH")
    assert result.switch_candidate is False


def test_controller_requires_two_consecutive_confirmations():
    controller = LanguageSwitchController(initial_language="HINGLISH", confirmations=2)
    first = controller.update("Please send the details tomorrow", "ENGLISH")
    assert first["switched"] is False
    assert first["current_language"] == "HINGLISH"
    second = controller.update("Please send the details tomorrow", "ENGLISH")
    assert second["switched"] is True
    assert second["current_language"] == "ENGLISH"


def test_neutral_resets_candidate_streak():
    controller = LanguageSwitchController(initial_language="HINGLISH", confirmations=2)
    controller.update("Please send the details tomorrow", "ENGLISH")
    neutral = controller.update("okay", "ENGLISH")
    assert neutral["switched"] is False
    assert neutral["streak"] == 0
    final = controller.update("Please send the details tomorrow", "ENGLISH")
    assert final["switched"] is False
