from typing import List, Dict, Any
import csv
import io
import os

def iterations_to_csv_string(iterations: List[Dict[str, Any]]) -> str:
    """
    Convert iterations list (list of dicts) to CSV string.
    Columns are: n, a, b, p, f(p), error
    """
    if not iterations:
        return ""
    fieldnames = ["n", "a", "b", "p", "f(p)", "error"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in iterations:
        # ensure values are plain scalars or empty
        clean_row = {k: (row.get(k) if row.get(k) is not None else "") for k in fieldnames}
        writer.writerow(clean_row)
    return buf.getvalue()

def save_iterations_to_csv(iterations: List[Dict[str, Any]], filepath: str) -> None:
    """
    Save iterations to a CSV file at filepath. Overwrites if exists.
    """
    csv_text = iterations_to_csv_string(iterations)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        f.write(csv_text)

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def pretty_format_number(x: float, digits: int = 8) -> str:
    """
    Format number for display: use digits decimal places unless large/small.
    """
    try:
        return f"{x:.{digits}g}"
    except Exception:
        return str(x)
