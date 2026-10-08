"""
Looking up chargers: by ID, and near a GPS position.

In reality the charger catalogue belongs to another Tap Electric service;
this module is the read-only view the scan service needs.
"""

import csv
import math
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app.storage.db import ChargerRow

EARTH_RADIUS_M = 6_371_000


@dataclass(frozen=True)
class Charger:
    charger_id: str
    evse_id: str
    latitude: float
    longitude: float


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two GPS points (haversine formula:
    the shortest distance over the surface of a sphere)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _key(evse_id: str) -> str:
    return evse_id.upper().replace("*", "")


class ChargerDirectory(Protocol):
    def get_by_evse(self, evse_id: str) -> Charger | None: ...

    def find_near(self, latitude: float, longitude: float, radius_m: float) -> list[Charger]: ...


class InMemoryChargerDirectory:
    def __init__(self, chargers: list[Charger] | None = None):
        self._chargers = list(chargers or [])

    @classmethod
    def from_csv(cls, path: str) -> "InMemoryChargerDirectory":
        """Load chargers from a CSV file with the columns
        charger_id,evse_id,latitude,longitude (handy for local testing)."""
        with open(path, newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            return cls([
                Charger(r["charger_id"], r["evse_id"], float(r["latitude"]), float(r["longitude"]))
                for r in rows
            ])

    def get_by_evse(self, evse_id: str) -> Charger | None:
        key = _key(evse_id)
        return next((c for c in self._chargers if _key(c.evse_id) == key), None)

    def find_near(self, latitude: float, longitude: float, radius_m: float) -> list[Charger]:
        return [c for c in self._chargers if distance_m(latitude, longitude, c.latitude, c.longitude) <= radius_m]


class SqlChargerDirectory:
    """Reads the `chargers` table.

    find_near first selects a square around the point (fast, can use an
    index), then removes the corners with the exact distance. In production
    PostGIS does this in one indexed query:
        WHERE ST_DWithin(location, ST_MakePoint(:lon, :lat)::geography, :radius)
    """

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def get_by_evse(self, evse_id: str) -> Charger | None:
        """Compare without asterisks: NL*TNM*E12345*1 == NLTNME123451.
        (In production, store this key in its own indexed column.)"""
        no_stars = func.replace(func.upper(ChargerRow.evse_id), "*", "")
        with self._session_factory() as session:
            row = session.scalar(select(ChargerRow).where(no_stars == _key(evse_id)))
            return _to_charger(row) if row else None

    def find_near(self, latitude: float, longitude: float, radius_m: float) -> list[Charger]:
        d_lat = math.degrees(radius_m / EARTH_RADIUS_M)
        d_lon = d_lat / max(math.cos(math.radians(latitude)), 1e-6)
        query = select(ChargerRow).where(
            ChargerRow.latitude.between(latitude - d_lat, latitude + d_lat),
            ChargerRow.longitude.between(longitude - d_lon, longitude + d_lon),
        )
        with self._session_factory() as session:
            rows = session.scalars(query).all()
        return [
            _to_charger(r) for r in rows
            if distance_m(latitude, longitude, r.latitude, r.longitude) <= radius_m
        ]


def _to_charger(row: ChargerRow) -> Charger:
    return Charger(row.charger_id, row.evse_id, row.latitude, row.longitude)
