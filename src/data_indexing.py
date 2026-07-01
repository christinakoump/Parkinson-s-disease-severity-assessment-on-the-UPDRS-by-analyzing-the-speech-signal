import os
import pandas as pd

def create_index(root_path, save_csv):
    folder_records = []

    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if not fname.endswith(".wav"):
                continue

            full_path = os.path.join(dirpath, fname)
            relative_path = os.path.relpath(full_path, root_path)
            parts = relative_path.split(os.sep)

            task = parts[0]
            speaker_id = fname.split("_")[0]

            path_lower = [p.lower() for p in parts]

            if any(x in p for p in path_lower for x in ["pd", "parkinson", "patologica"]):
                label = "PD"
            elif any(x in p for x in ["control", "hc", "normal"]):
                label = "HC"
            else:
                label = "Unknown"

            folder_records.append({
                "Task": task,
                "Label": label,
                "Speaker_ID": speaker_id,
                "Relative Path": relative_path,
                "Full Path": full_path,
            })

    df = pd.DataFrame(folder_records)
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv, index=False)

    return df
