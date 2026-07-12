# This file computes the reference coordinates for each station in the Arome dataset, based on the most frequently occurring coordinates for each station ID. 
# It outputs a CSV file with the station ID, reference longitude and latitude, number of occurrences of the reference coordinates, total number of occurrences for that station, 
# and the purity (proportion of occurrences that match the reference coordinates). 
# This is part of the cleaning process to ensure that each station has a consistent set of coordinates.

import csv
from collections import Counter, defaultdict
from pathlib import Path
 
# Params
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "Arome"
OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "stations_reference.csv"


def main():
    coord_counts = defaultdict(Counter)
    n_total_lines = 0
    n_18 = 0

    with open(INPUT_FILE, "r", errors="replace") as f:
        for raw in f:
            n_total_lines += 1
            tokens = raw.split()

            if len(tokens) != 18:
                continue
            n_18 += 1

            station_id = tokens[0]
            lon, lat = tokens[2], tokens[3]
            coord_counts[station_id][(lon, lat)] += 1

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lon", "lat", "n_occurrences", "n_total", "purity"])

        for station_id in sorted(coord_counts):
            counter = coord_counts[station_id]
            (lon, lat), n_occurrences = counter.most_common(1)[0]
            n_total = sum(counter.values())
            purity = n_occurrences / n_total
            writer.writerow([station_id, lon, lat, n_occurrences, n_total, f"{purity:.4f}"])

    print("=== Report ===")
    print(f"Lines read (total)              : {n_total_lines}")
    print(f"Lines with 18 fields (with id)  : {n_18}")
    print(f"Distinct station IDs found      : {len(coord_counts)}")
    print(f"Output file                       : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()