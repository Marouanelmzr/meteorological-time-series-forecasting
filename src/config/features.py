

# METADATA (NOT TRAINING FEATURES)

METADATA_COLUMNS = [
    "station_id",
    "icao",
    "station_name",
    "run_time",
    "valid_time",
]

# STATION FEATURES

STATION_FEATURES = [
    "arome_lon",
    "arome_lat",
    "station_lon",
    "station_lat",
    "elevation_m",
    "distance_km",
]

# TIME FEATURES

TIME_FEATURES = [
    "lead_time",
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
]

# AROME WIND FEATURES

AROME_WIND10 = [
    "u10",
    "v10",
    "arome_wind10_speed",
    "arome_wind10_dir",
]

AROME_WIND850 = [
    "u850",
    "v850",
    "arome_wind850_speed",
    "arome_wind850_dir",
]

AROME_WIND950 = [
    "u950",
    "v950",
    "arome_wind950_speed",
    "arome_wind950_dir",
]

# AROME ATMOSPHERIC FEATURES

AROME_ATMOSPHERE = [
    "t2m",
    "rh2m",
    "psurf",
    "pblh",
    "tke20m",
    "edr20m",
]

# AROME GUST FEATURES

AROME_GUST = [
    "u_gust60",
    "v_gust60",
    "arome_gust60_speed",
    "arome_gust60_dir",
]

# DERIVED FEATURES

DERIVED_FEATURES = [
    "wind_shear_950_10",
]

# METAR FEATURES
# (Only useful for experiments using observations)

METAR_FEATURES = [
    "metar_source",
    "metar_wind_dir_deg",
    "metar_wind_speed_ms",
    "metar_visibility_m",
    "metar_temp_c",
    "metar_dewpoint_c",
    "metar_slp_hpa",
    "metar_ceiling_m",
    "metar_clouds",
    "metar_present_weather",
    "metar_avwx_gust_ms",
    "metar_gust_mismatch",
]

# TARGETS

CLASSIFICATION_TARGET = "has_gust"

REGRESSION_TARGET = "gust_speed_ms"

# AUXILIARY COLUMNS (NOT FEATURES)

AUXILIARY_COLUMNS = [
    "gust_reclassified_noise",
]

# FEATURE SETS

AROME_FEATURES = (
    STATION_FEATURES
    + TIME_FEATURES
    + AROME_WIND10
    + AROME_WIND850
    + AROME_WIND950
    + AROME_ATMOSPHERE
    + AROME_GUST
    + DERIVED_FEATURES
)

ALL_FEATURES = (
    AROME_FEATURES
    + METAR_FEATURES
)

# COMPLETE DATASET COLUMN ORDER

ALL_COLUMNS = (
    METADATA_COLUMNS
    + STATION_FEATURES
    + TIME_FEATURES
    + AROME_WIND10
    + AROME_WIND850
    + AROME_WIND950
    + AROME_ATMOSPHERE
    + AROME_GUST
    + DERIVED_FEATURES
    + METAR_FEATURES
    + [
        CLASSIFICATION_TARGET,
        REGRESSION_TARGET,
    ]
    + AUXILIARY_COLUMNS
)