# This file removes the station field (index 0) from the Arome dataset, and checks that all lines have the expected number of fields (17 after dropping the station field).
# It is part of the cleaning process, and is used to produce a clean CSV file with the expected structure. ( See the notebook 01_exploration.ipynb for more details on the structure of the Arome dataset.)

import csv
from pathlib import Path
 
# Params
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_CLEANED = PROJECT_ROOT / "data" / "cleaned"
DATA_CLEANED.mkdir(parents=True, exist_ok=True)  # create it if it doesn't exist yet
 
INPUT_FILE = DATA_RAW / "Arome"
OUTPUT_FILE = DATA_CLEANED / "Arome_structure_clean.csv"
REJECTED_FILE = DATA_CLEANED / "rejected_structure.tsv"

COLUMNS = [
    "datetime", "lon", "lat",
    "u10", "v10", "t2m", "rh2m",
    "u850", "v850", "u950", "v950",
    "psurf", "u_gust60", "v_gust60",
    "tke20m", "edr20m", "pblh",
]  # 17 fields, station (index 0) is dropped

N_EXPECTED = len(COLUMNS)  # 17


def normalize(tokens):
    """Ramene une liste de tokens a la forme a 17 champs, ou None si rejetee.
    Retourne (normalized_tokens, reason) -- reason est None si acceptee."""
    n = len(tokens)

    if n == 16:
        return None, "16_fields_missing_date"

    if n == 17:
        return tokens, None

    if n == 18:
        return tokens[1:], None  # We drop the station field (index 0)

    return None, f"unexpected_field_count_{n}"


def main():
    n_total = 0
    n_16 = 0
    n_other_bad = 0
    n_from_17 = 0
    n_from_18 = 0
    n_duplicates = 0

    seen = set()
    rejected_f = open(REJECTED_FILE, "w", newline="")
    rej_writer = csv.writer(rejected_f, delimiter="\t")
    rej_writer.writerow(["line_no", "n_fields", "reason", "raw_preview"])

    out_f = open(OUTPUT_FILE, "w", newline="")
    out_writer = csv.writer(out_f)
    out_writer.writerow(COLUMNS)

    with open(INPUT_FILE, "r", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            n_total += 1
            tokens = raw.split()

            normalized, reason = normalize(tokens)

            if reason is not None:
                if reason == "16_fields_missing_date":
                    n_16 += 1
                else:
                    n_other_bad += 1
                rej_writer.writerow([line_no, len(tokens), reason, raw.strip()[:120]])
                continue

            if len(tokens) == 17:
                n_from_17 += 1
            else:
                n_from_18 += 1

            key = tuple(normalized)
            if key in seen:
                n_duplicates += 1
                rej_writer.writerow([line_no, len(tokens), "duplicate_after_station_drop", raw.strip()[:120]])
                continue
            seen.add(key)

            out_writer.writerow(normalized)

    rejected_f.close()
    out_f.close()

    n_kept = len(seen)

    print("=== Report ===")
    print(f"Lines read                          : {n_total}")
    print(f"Rejected (16 fields, missing date)  : {n_16}")
    print(f"Rejected (unexpected field count) : {n_other_bad}")
    print(f"Issues with 17-field lines (station already dropped) : {n_from_17}")
    print(f"Issues with 18-field lines (station dropped)       : {n_from_18}")
    print(f"Deduplicates             : {n_duplicates}")
    print(f"Lines kept in the output      : {n_kept}")
    print(f"Output file                     : {OUTPUT_FILE}")
    print(f"Rejection log                        : {REJECTED_FILE}")


if __name__ == "__main__":
    main()