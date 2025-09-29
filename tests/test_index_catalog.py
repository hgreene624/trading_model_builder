from src import storage
from src.data import universe


def test_list_index_cache_exposes_sample_universes():
    indexes = storage.list_index_cache()
    assert "S&P 100 Sample" in indexes
    assert "Nasdaq Growth 50" in indexes


def test_load_index_members_normalizes_symbols():
    payload = storage.load_index_members("S&P 100 Sample")
    assert payload is not None
    symbols = payload.get("symbols")
    assert isinstance(symbols, list)
    assert "BRK-B" in symbols  # normalized from BRK.B in file
    members = payload.get("members")
    assert members and all("symbol" in row for row in members)
    meta = payload.get("meta", {})
    assert meta.get("source_type") == "aggregate"
    assert meta.get("source_path", "").endswith("indexes.json")


def test_universe_module_reads_index_file():
    idx_map = universe.load_indexes()
    assert "Dividend Achievers" in idx_map
    assert "KO" in idx_map["Dividend Achievers"]
    available = universe.available_universes()
    assert available == sorted(idx_map.keys())
    achievers = universe.get_universe("Dividend Achievers")
    assert set(achievers) == set(idx_map["Dividend Achievers"])
