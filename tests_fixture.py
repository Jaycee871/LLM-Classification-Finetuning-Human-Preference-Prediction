"""Small fake competition fixture, no copyrighted data."""
import pandas as pd


def synthetic_data():
    rows = []
    for cls in range(3):
        for j in range(12):
            rows.append({
                "id": 12 * cls + j,
                "prompt": '["Explain this artificial example"]',
                "response_a": '["Thorough structured explanation of dummy topic %s"]' % j
                  if cls == 0 else '["Brief reply %s"]' % j,
                "response_b": '["Thorough structured explanation of dummy topic %s"]' % j
                  if cls == 1 else '["Brief reply %s"]' % j,
                "winner_model_a": int(cls == 0),
                "winner_model_b": int(cls == 1),
                "winner_tie": int(cls == 2),
            })
    return pd.DataFrame(rows)
