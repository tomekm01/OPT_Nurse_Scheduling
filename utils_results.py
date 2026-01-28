# utils_results.py

import csv, os, json

def json_params(d: dict) -> str:
    return json.dumps(d, sort_keys=True)

def append_row(csv_path: str, row: dict) -> None:
    folder = os.path.dirname(csv_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
