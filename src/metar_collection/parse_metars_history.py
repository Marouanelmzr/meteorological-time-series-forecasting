# This script parses the NOAA METAR bulletin files (metars_1.csv and metars_backup.csv)

import re
from pathlib import Path

import avwx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BULLETIN_FILES = [
    ROOT / "data" / "raw" / "metars_1.csv",
    ROOT / "data" / "raw" / "metars_backup.csv",
]
NOAA_METAR_FILE = ROOT / "data" / "raw" / "METAR_2021_2025.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "METAR_bulletin_2021_2025.csv"

# Captures: sid, icao, ddhhmm (day/hour/minute), dated (YYYY-MM-DD), then the
# message body up to ";source;{...}" (non-greedy, spans embedded \r\r breaks).
RECORD_PATTERN = re.compile(
    rb"\d+;SAMC40_([A-Z]{4})_(\d{6})\.\d+\.X;\d+;\d+;(\d{4}-\d{2}-\d{2});"
    rb"SAMC40 [A-Z]{4} \d{6}\r\r\n(.*?);\w+;\{[^}]*\}",
    re.DOTALL,
)


def load_station_metadata(path: Path) -> dict:
    df = pd.read_csv(
        path,
        usecols=["icao", "station", "latitude", "longitude", "elevation_m", "name"],
        dtype={"icao": str, "station": str},
        low_memory=False,
    )
    df = df.dropna(subset=["icao"]).drop_duplicates(subset=["icao"])
    return df.set_index("icao").to_dict(orient="index")


def extract_records(path: Path):
    with open(path, "rb") as f:
        data = f.read()

    for match in RECORD_PATTERN.finditer(data):
        icao_b, ddhhmm_b, dated_b, message_b = match.groups()
        icao = icao_b.decode()
        ddhhmm = ddhhmm_b.decode()
        dated = dated_b.decode()

        hour, minute = ddhhmm[2:4], ddhhmm[4:6]
        try:
            timestamp = pd.Timestamp(f"{dated} {hour}:{minute}:00")
        except ValueError:
            continue

        message = message_b.decode(errors="replace").replace("\r", " ").strip()
        message = re.sub(r"^(METAR|SPECI)\s+", "", message).strip()
        if not message:
            continue

        yield icao, timestamp, message


def parse_report(icao: str, raw: str) -> dict | None:
    try:
        report = avwx.Metar(icao)
        if not report.parse(raw):
            return None
        d = report.data
    except Exception:
        return None

    def num(field):
        return field.value if field is not None else None

    wind_speed_kt = num(d.wind_speed)
    gust_kt = num(d.wind_gust)

    clouds = [c for c in (d.clouds or [])]
    ceiling_m = None
    for c in clouds:
        if c.type in ("BKN", "OVC") and c.base is not None:
            candidate = c.base * 100 * 0.3048  # hundreds of ft -> m
            ceiling_m = candidate if ceiling_m is None else min(ceiling_m, candidate)

    return {
        "wind_dir_deg": num(d.wind_direction),
        "wind_speed_ms": wind_speed_kt * 0.514444 if wind_speed_kt is not None else None,
        "visibility_m": num(d.visibility),
        "temp_c": num(d.temperature),
        "dewpoint_c": num(d.dewpoint),
        "slp_hpa": num(d.altimeter),  # QNH proxy, see module docstring
        "ceiling_m": ceiling_m,
        "has_gust": int(gust_kt is not None),
        "gust_speed_ms": gust_kt * 0.514444 if gust_kt is not None else None,
        "clouds": "; ".join(f"{c.type}{c.base or ''}" for c in clouds) or None,
        "present_weather": "; ".join(w.value for w in (d.wx_codes or [])) or None,
    }


def main():
    metadata = load_station_metadata(NOAA_METAR_FILE)

    rows = []
    seen = set()  # dedup (icao, timestamp) across both files
    parsed_ok = 0
    parse_failed = 0

    for path in BULLETIN_FILES:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        for icao, timestamp, message in extract_records(path):
            key = (icao, timestamp)
            if key in seen:
                continue
            seen.add(key)

            decoded = parse_report(icao, message)
            if decoded is None:
                parse_failed += 1
                continue
            parsed_ok += 1

            meta = metadata.get(icao, {})
            rows.append({
                "icao": icao,
                "station": meta.get("station"),
                "date": timestamp,
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "elevation_m": meta.get("elevation_m"),
                "name": meta.get("name"),
                **decoded,
                "avwx_gust_ms": decoded["gust_speed_ms"],
                "gust_mismatch": False,
            })

    out = pd.DataFrame(rows)

    # Match METAR_2021_2025.csv column order exactly
    columns = [
        "icao", "station", "date", "latitude", "longitude", "elevation_m", "name",
        "wind_dir_deg", "wind_speed_ms", "visibility_m", "temp_c", "dewpoint_c",
        "slp_hpa", "ceiling_m", "has_gust", "gust_speed_ms", "clouds",
        "present_weather", "avwx_gust_ms", "gust_mismatch",
    ]
    out = out[columns].sort_values(["icao", "date"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"Records parsed successfully: {parsed_ok:,}")
    print(f"Records failed to parse: {parse_failed:,}")
    print(f"Rows without station metadata match (icao not in METAR_2021_2025.csv): "
          f"{out['station'].isna().sum():,}")
    print(f"Stations: {out['icao'].nunique()}")
    print(f"Saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()