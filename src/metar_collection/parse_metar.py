# This script parses NOAA ISD 2021-2025 CSV data into a more usable METAR format, 
# extracting core weather fields and enriching with cloud and present weather information using the avwx library.

import re
from pathlib import Path

import avwx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "raw" / "NOAA_ISD_2021_2025.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "METAR_2021_2025.csv"

REM_PATTERN = re.compile(r"^MET\d{3}(.*)$")


# Primary source: manual decoding of structured ISD columns

def _split(series: pd.Series, n_parts: int) -> list[pd.Series]:
    parts = series.fillna("").str.split(",", n=n_parts - 1, expand=True)
    parts = parts.reindex(columns=range(n_parts))
    return [parts[i] for i in range(n_parts)]


def manual_decode(df: pd.DataFrame) -> pd.DataFrame:
    direction, _qdir, _wtype, speed, _qspeed = _split(df["WND"], 5)
    direction_num = pd.to_numeric(direction, errors="coerce").where(lambda s: s != 999, pd.NA)
    speed_ms = (pd.to_numeric(speed, errors="coerce") / 10.0).where(lambda s: s != 999.9, pd.NA)

    tmp_val, _tmp_q = _split(df["TMP"], 2)
    temp_num = (pd.to_numeric(tmp_val, errors="coerce") / 10.0).where(lambda s: s != 999.9, pd.NA)

    dew_val, _dew_q = _split(df["DEW"], 2)
    dew_num = (pd.to_numeric(dew_val, errors="coerce") / 10.0).where(lambda s: s != 999.9, pd.NA)

    slp_val, _slp_q = _split(df["SLP"], 2)
    slp_num = (pd.to_numeric(slp_val, errors="coerce") / 10.0).where(lambda s: s != 9999.9, pd.NA)

    vis_val, _vqd, _vvar, _vqvar = _split(df["VIS"], 4)
    vis_num = pd.to_numeric(vis_val, errors="coerce").where(lambda s: s != 999999, pd.NA)

    cig_val, _cq, _cdet, _ccavok = _split(df["CIG"], 4)
    cig_num = pd.to_numeric(cig_val, errors="coerce").where(lambda s: s != 99999, pd.NA)

    # OC1 = Wind-Gust-observation: "speed_rate,quality_code", m/s x10, missing = 9999
    if "OC1" in df.columns:
        gust_val, _gust_q = _split(df["OC1"], 2)
        gust_ms = (pd.to_numeric(gust_val, errors="coerce") / 10.0).where(lambda s: s != 999.9, pd.NA)
    else:
        gust_ms = pd.Series(pd.NA, index=df.index)

    has_gust = gust_ms.notna().astype(int)

    return pd.DataFrame({
        "wind_dir_deg": direction_num,
        "wind_speed_ms": speed_ms,
        "visibility_m": vis_num,
        "temp_c": temp_num,
        "dewpoint_c": dew_num,
        "slp_hpa": slp_num,
        "ceiling_m": cig_num,
        "has_gust": has_gust,
        "gust_speed_ms": gust_ms,
    })


# Enrichment only: raw METAR via avwx (clouds, present weather) 

def extract_raw_metar(rem: str) -> str | None:
    if not isinstance(rem, str) or not rem.strip():
        return None

    match = REM_PATTERN.match(rem.strip())
    if not match:
        return None

    text = match.group(1).strip().rstrip("=").strip()
    text = re.sub(r"^(METAR|SPECI)\s+", "", text)
    return text or None


def enrich_with_avwx(icao: str, raw: str) -> dict:
    empty = {"clouds": None, "present_weather": None, "avwx_gust_kt": None}
    try:
        report = avwx.Metar(icao)
        if not report.parse(raw):
            return empty
        d = report.data
    except Exception:
        return empty

    clouds = "; ".join(f"{c.type}{c.base or ''}" for c in (d.clouds or [])) or None
    present_wx = "; ".join(w.value for w in (d.wx_codes or [])) or None
    avwx_gust = d.wind_gust.value if d.wind_gust is not None else None

    return {"clouds": clouds, "present_weather": present_wx, "avwx_gust_kt": avwx_gust}


def main():
    raw = pd.read_csv(INPUT_FILE, dtype=str, low_memory=False)

    base = pd.DataFrame({
        "icao": raw["icao"],
        "station": raw.get("STATION"),
        "date": pd.to_datetime(raw["DATE"], errors="coerce"),
        "latitude": pd.to_numeric(raw.get("LATITUDE"), errors="coerce"),
        "longitude": pd.to_numeric(raw.get("LONGITUDE"), errors="coerce"),
        "elevation_m": pd.to_numeric(raw.get("ELEVATION"), errors="coerce"),
        "name": raw.get("NAME"),
    })

    core = manual_decode(raw)

    enrichment_rows = []
    enriched_ok = 0
    for icao, rem in zip(raw["icao"], raw.get("REM", pd.Series([None] * len(raw)))):
        raw_metar = extract_raw_metar(rem)
        result = enrich_with_avwx(icao, raw_metar) if raw_metar else {
            "clouds": None, "present_weather": None, "avwx_gust_kt": None
        }
        if result["clouds"] is not None or result["present_weather"] is not None:
            enriched_ok += 1
        enrichment_rows.append(result)

    enrichment = pd.DataFrame(enrichment_rows)

    out = pd.concat([base, core, enrichment], axis=1)
    out = out.dropna(subset=["date"]).sort_values(["icao", "date"])

    # Gust consistency check: manual OC1 says "no gust" but avwx found a gust group
    # in the raw METAR text, or vice versa. Doesn't change gust_speed_ms (which stays
    # OC1-based for consistency), just flags rows worth a manual look. Kept as columns
    # (not dropped) so mismatches can be diagnosed later without re-running the parser.
    out["avwx_gust_ms"] = out["avwx_gust_kt"] * 0.514444
    out["gust_mismatch"] = (out["has_gust"] == 1) != out["avwx_gust_ms"].notna()
    out["gust_mismatch"] &= out["avwx_gust_ms"].notna() | out["gust_speed_ms"].notna()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns=["avwx_gust_kt"]).to_csv(OUTPUT_FILE, index=False)

    mismatch = out["gust_mismatch"]
    print(f"Rows total: {len(out):,}")
    print(f"Core fields decoded (manual, 100% coverage): {len(out):,}")
    print(f"Rows with gust (OC1): {out['has_gust'].sum():,} ({out['has_gust'].mean() * 100:.2f}%)")
    print(f"OC1 vs avwx gust mismatches: {mismatch.sum():,}")
    print(f"Enriched with clouds/present weather (avwx): {enriched_ok:,} ({enriched_ok / len(out) * 100:.1f}%)")
    print(f"Stations: {out['icao'].nunique()}")
    print(f"Saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()