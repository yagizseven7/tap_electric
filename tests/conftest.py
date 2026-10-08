"""
Pytest fixtures shared by several test files.

pytest finds this file automatically. A fixture is a function that
prepares something a test needs; a test asks for it by naming it as a
parameter, e.g. `def test_x(synthetic_world): ...`.

scope="session" means it is built once for the whole test run and shared,
which matters for things that are slow to create.
"""

from dataclasses import dataclass

import pytest

from app.storage.chargers import Charger
from app.storage.object_store import InMemoryObjectStore
from app.storage.repository import InMemoryScanRepository
from app.training.synthetic import SyntheticScan, generate, store_scans


@dataclass
class SyntheticWorld:
    chargers: list[Charger]
    scans: list[SyntheticScan]
    repo: InMemoryScanRepository
    store: InMemoryObjectStore


@pytest.fixture(scope="session")
def synthetic_world() -> SyntheticWorld:
    """60 chargers (in stations of 1-4 connectors), 2 photos each, already
    stored through the repository: a small, realistic world for tests."""
    chargers, scans = generate(n_chargers=60, scans_per_charger=2, seed=11)
    repo, store = InMemoryScanRepository(), InMemoryObjectStore()
    store_scans(scans, repo, store)
    return SyntheticWorld(chargers, scans, repo, store)
