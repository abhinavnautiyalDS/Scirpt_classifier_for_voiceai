import pandas as pd

from src.prepare_data import split_data


def test_grouped_split_has_no_call_leakage():
    rows = []
    labels = ["HINGLISH", "ENGLISH", "NEUTRAL", "UNCERTAIN"]
    for call_id in range(1, 22):
        for label in labels:
            rows.append({
                "call_id": call_id,
                "text": f"sample {call_id} {label}",
                "label": label,
            })
    df = pd.DataFrame(rows)
    train, validation, test = split_data(df, seed=42)
    train_calls = set(train.call_id)
    validation_calls = set(validation.call_id)
    test_calls = set(test.call_id)
    assert not (train_calls & validation_calls)
    assert not (train_calls & test_calls)
    assert not (validation_calls & test_calls)
    assert set(train.label) == set(labels)
    assert set(validation.label) == set(labels)
    assert set(test.label) == set(labels)
