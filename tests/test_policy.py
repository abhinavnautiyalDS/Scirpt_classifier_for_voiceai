from src.predict import LABELS, NEUTRAL_PHRASES


def test_label_inventory_is_fixed():
    assert set(LABELS) == {"HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"}


def test_neutral_phrase_list_contains_common_acknowledgements():
    assert "hello" in NEUTRAL_PHRASES
    assert "okay" in NEUTRAL_PHRASES
    assert "नमस्ते" in NEUTRAL_PHRASES
