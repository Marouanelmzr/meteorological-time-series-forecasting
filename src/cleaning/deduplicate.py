# This file removes duplicate lines from the filtered Arome dataset, keeping only the line with the smallest distance for each (id, datetime) pair.
# It outputs a CSV file with the cleaned lines, and prints a report of the number of lines read, duplicates removed, and lines kept.

import csv
from pathlib import Path
 
# Params
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "Arome_filtered.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "Arome_clean_final.csv"


def main():
    with open(INPUT_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_total = len(rows)

    best = {}  # (id, datetime) -> row
    for row in rows:
        key = (row["id"], row["datetime"])
        dist = float(row["distance_km"])

        if key not in best or dist < float(best[key]["distance_km"]): # We keep the row with the smallest distance for each (id, datetime) pair.
            best[key] = row

    n_kept = len(best)
    n_duplicates = n_total - n_kept

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in best.values():
            writer.writerow(row)

    print("=== Report deduplicate ===")
    print(f"Lines read           : {n_total}")
    print(f"Duplicates (id, datetime) removed : {n_duplicates}")
    print(f"Lines kept           : {n_kept}")
    print(f"Output file          : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()