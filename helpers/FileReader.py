import pandas as pd

def extract_exact_fields_from_excel(filepath):
    df = pd.read_excel(filepath, engine="openpyxl")
    df.dropna(how='all', inplace=True)

    label_map = {
        "target audience": "target_audience",
        "tone/voice & vocabulary": "tone_of_voice_and_vocab",
        "important notes": "important_notes",
        "ce/cs": "ce_cs"
    }

    result = {}

    for i in range(len(df)):
        label = df.iloc[i, 0]
        if isinstance(label, str):
            label_clean = label.strip().lower().replace("\n", " ")
            for excel_label, json_key in label_map.items():
                # Match even if there are extra spaces
                if excel_label.replace(" ", "") in label_clean.replace(" ", ""):
                    value = df.iloc[i, 2]
                    if isinstance(value, str):
                        result[json_key] = value.strip()

    return result