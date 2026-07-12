# This file filters the Arome dataset to keep only the lines that are within a certain distance threshold from the nearest reference station.
# It outputs a CSV file with the filtered lines, including the nearest station ID and the distance to that station. 
# Lines that are rejected (i.e., those that are too far from any reference station) are logged in a separate TSV file.

import csv
import math
from pathlib import Path
 
# Params
 
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "Arome_structure_clean.csv"
STATIONS_FILE = PROJECT_ROOT / "data" / "cleaned" / "stations_reference.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "cleaned" / "Arome_filtered.csv"
REJECTED_FILE = PROJECT_ROOT / "data" / "cleaned" / "Arome_rejected_distance.tsv"

THRESHOLD_KM = 10.0
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def load_stations(path):
    stations = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            stations.append((row["id"], float(row["lon"]), float(row["lat"])))
    return stations


def nearest_station(lon, lat, stations):
    best_id, best_dist = None, None
    for station_id, s_lon, s_lat in stations:
        d = haversine_km(lon, lat, s_lon, s_lat)
        if best_dist is None or d < best_dist:
            best_id, best_dist = station_id, d
    return best_id, best_dist


def main():
    stations = load_stations(STATIONS_FILE)
    print(f"Stations de reference chargees : {len(stations)}")

    n_total = 0
    n_rejected = 0
    n_kept = 0

    with open(INPUT_FILE, "r", newline="") as fin, \
         open(OUTPUT_FILE, "w", newline="") as fout, \
         open(REJECTED_FILE, "w", newline="") as frej:

        reader = csv.DictReader(fin)
        var_columns = [c for c in reader.fieldnames if c not in ("datetime", "lon", "lat")]

        writer = csv.writer(fout)
        writer.writerow(["id", "datetime", "lon", "lat"] + var_columns + ["distance_km"])

        rej_writer = csv.writer(frej, delimiter="\t")
        rej_writer.writerow(["datetime", "lon", "lat", "nearest_id", "distance_km"])

        for row in reader:
            n_total += 1
            lon, lat = float(row["lon"]), float(row["lat"])

            best_id, best_dist = nearest_station(lon, lat, stations)

            if best_dist > THRESHOLD_KM:
                n_rejected += 1
                rej_writer.writerow([row["datetime"], lon, lat, best_id, f"{best_dist:.3f}"])
                continue

            n_kept += 1
            writer.writerow(
                [best_id, row["datetime"], lon, lat]
                + [row[c] for c in var_columns]
                + [f"{best_dist:.3f}"]
            )

            if n_total % 200000 == 0:
                print(f"... {n_total} lines treated")

    print()
    print("=== Report filter_by_distance ===")
    print(f"Lines read                        : {n_total}")
    print(f"Distance threshold                : {THRESHOLD_KM} km")
    print(f"Lines rejected (> threshold)       : {n_rejected}")
    print(f"Lines kept                         : {n_kept}")
    print(f"Output file                        : {OUTPUT_FILE}")
    print(f"Log rejected                       : {REJECTED_FILE}")


if __name__ == "__main__":
    main()