import pandas as pd
import re
def extract_exact_fields_from_excel(filepath):
    def normalize_label(label):
        # Lowercase and remove all non-alphabet characters (e.g., spaces, slashes, ampersands)
        return re.sub(r'[^a-z]', '', label.lower())

    df = pd.read_excel(filepath, engine="openpyxl")
    df.dropna(how='all', inplace=True)

    # df = pd.read_excel(filepath, engine="openpyxl")
    # df.dropna(how='all', inplace=True)

    label_map = {
        "target audience": "target_audience",
        "tone/voice & vocabulary": "tone_of_voice_and_vocab",
        "Tone & Vocabulary": "tone_of_voice_and_vocab",
        "important notes": "important_notes",
        "ce/cs": "ce_cs"
    }

    result = {}

    for i in range(len(df)):
        label = df.iloc[i, 0]
        if isinstance(label, str):
            normalized_label = normalize_label(label)
            for excel_label, json_key in label_map.items():
                if normalize_label(excel_label) in normalized_label:
                    value = df.iloc[i, 2]
                    if isinstance(value, str):
                        result[json_key] = value.strip()

    return result