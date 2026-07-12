from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ISD_HISTORY_FILE = ROOT / "data" / "cleaned" / "isd-history.csv"
STATIONS_FILE = ROOT / "data" / "cleaned" / "stations_reference.csv"

COORD_TOLERANCE = 0.3  # degrees

# Load datasets
isd = pd.read_csv(ISD_HISTORY_FILE, dtype=str)
stations = pd.read_csv(STATIONS_FILE, dtype={"id": str})

# Clean NOAA column names
isd.columns = isd.columns.str.strip()

# Convert coordinates
isd["LAT"] = pd.to_numeric(isd["LAT"], errors="coerce")
isd["LON"] = pd.to_numeric(isd["LON"], errors="coerce")

# Remove surrounding whitespace
isd["USAF"] = isd["USAF"].str.strip()
isd["ICAO"] = isd["ICAO"].fillna("").str.strip()

results = []

for _, station in stations.iterrows():

    wmo_id = station["id"]
    lon = float(station["lon"])
    lat = float(station["lat"])

    # Moroccan stations are stored as WMO + "0"
    usaf = f"{wmo_id}0"

    match = isd[isd["USAF"] == usaf]

    if match.empty:
        results.append({
            "id": wmo_id,
            "found": False,
            "coord_match": None,
            "icao": None,
            "station_name": None,
            "lat_arome": lat,
            "lon_arome": lon,
            "lat_isd": None,
            "lon_isd": None,
            "begin": None,
            "end": None,
        })
        continue

    row = match.iloc[0]

    coord_match = (
        abs(row["LAT"] - lat) <= COORD_TOLERANCE
        and abs(row["LON"] - lon) <= COORD_TOLERANCE
    )

    results.append({
        "id": wmo_id,
        "found": True,
        "coord_match": coord_match,
        "icao": row["ICAO"],
        "station_name": row["STATION NAME"],
        "lat_arome": lat,
        "lon_arome": lon,
        "lat_isd": row["LAT"],
        "lon_isd": row["LON"],
        "begin": row["BEGIN"],
        "end": row["END"],
    })

report = pd.DataFrame(results)

print(report)

print(f"\nFound: {report['found'].sum()} / {len(stations)}")

missing = report.loc[~report["found"], "id"]

if not missing.empty:
    print("\nMissing stations:")
    print(missing.tolist())

mismatch = report[(report["found"]) & (report["coord_match"] == False)]

if not mismatch.empty:
    print("\nCoordinate mismatches:")
    print(
        mismatch[
            [
                "id",
                "icao",
                "lat_arome",
                "lon_arome",
                "lat_isd",
                "lon_isd",
            ]
        ]
    )

report.to_csv(
    ROOT / "data" / "cleaned" / "isd_history_station_check.csv",
    index=False,
)

print("\nReport written to data/cleaned/isd_history_station_check.csv")