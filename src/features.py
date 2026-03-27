"""
features.py
-----------
Feature engineering for SF crime category prediction.

Transforms raw SFPD incident data into a numeric feature matrix
suitable for scikit-learn / XGBoost pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Downtown SF reference point for distance feature
_DOWNTOWN_LAT = 37.7749
_DOWNTOWN_LON = -122.4194
_EARTH_RADIUS_KM = 6371.0

# Coordinate bounding box for SF (removes the Y=90 sentinel and outliers)
_LAT_MIN, _LAT_MAX = 37.6, 38.0
_LON_MIN, _LON_MAX = -123.0, -122.0

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
}

HOUR_BINS = [0, 6, 12, 17, 21, 24]
HOUR_LABELS = ["Late Night", "Morning", "Afternoon", "Evening", "Night"]

NUMERIC_FEATURES = ["Hour", "DayOfWeek_Num", "IsWeekend", "Month", "DistanceFromDowntown_km", "GeoCluster"]
CATEGORICAL_FEATURES = ["PdDistrict", "HourBlock", "Season", "DayOfWeek"]


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with invalid GPS coordinates (sentinel Y=90, out-of-SF points)."""
    mask = (
        (df["Y"] > _LAT_MIN) & (df["Y"] < _LAT_MAX) &
        (df["X"] > _LON_MIN) & (df["X"] < _LON_MAX)
    )
    n_removed = (~mask).sum()
    if n_removed:
        print(f"Removed {n_removed:,} records with invalid coordinates.")
    return df[mask].copy()


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Dates column and derive time-based features."""
    df = df.copy()
    df["Dates"] = pd.to_datetime(df["Dates"])
    df["Year"] = df["Dates"].dt.year
    df["Month"] = df["Dates"].dt.month
    df["Hour"] = df["Dates"].dt.hour
    df["DayOfWeek_Num"] = df["Dates"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek_Num"].isin([5, 6]).astype(int)
    df["HourBlock"] = pd.cut(df["Hour"], bins=HOUR_BINS, labels=HOUR_LABELS, right=False)
    df["Season"] = df["Month"].map(SEASON_MAP)
    return df


def _haversine_km(lat: float, lon: float, ref_lat: float, ref_lon: float) -> float:
    """Haversine great-circle distance in km between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat, ref_lat, lon, ref_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def add_spatial_features(df: pd.DataFrame, geo_kmeans: KMeans | None = None, n_clusters: int = 15) -> tuple[pd.DataFrame, KMeans]:
    """
    Add DistanceFromDowntown_km and KMeans-derived GeoCluster.

    If geo_kmeans is None, fits a new KMeans on df's coordinates and returns it
    alongside the augmented DataFrame so the same model can be applied to test data.
    """
    df = df.copy()
    df["DistanceFromDowntown_km"] = df.apply(
        lambda r: _haversine_km(r["Y"], r["X"], _DOWNTOWN_LAT, _DOWNTOWN_LON), axis=1
    )
    if geo_kmeans is None:
        geo_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        geo_kmeans.fit(df[["X", "Y"]])
    df["GeoCluster"] = geo_kmeans.predict(df[["X", "Y"]]).astype(str)
    return df, geo_kmeans


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select and return the final feature columns used by the model."""
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
