from tsdr.core.directory.favorites import FavoritesStore
from tsdr.core.directory.model import PublicDevice


def _device(device_id: str = "spyserver:9.9.9.9:5555", **overrides) -> PublicDevice:
    base = {
        "source": "spyserver",
        "id": device_id,
        "name": "Test Receiver",
        "host": "9.9.9.9",
        "port": 5555,
        "url": "sdr://9.9.9.9:5555",
        "location": "Testville",
        "freq_min": 0.0,
        "freq_max": 1700000000.0,
        "sample_rate": 768000.0,
        "usable": True,
    }
    base.update(overrides)
    return PublicDevice(**base)


def test_round_trip_persists_across_stores() -> None:
    store = FavoritesStore()
    store.add(_device("spyserver:1.1.1.1:5555", host="1.1.1.1"))
    store.add(_device("kiwisdr:http://k.example:8073", source="kiwisdr"))

    reloaded = FavoritesStore()
    reloaded.load()
    ids = {f.id for f in reloaded.all()}
    assert ids == {"spyserver:1.1.1.1:5555", "kiwisdr:http://k.example:8073"}


def test_remove_persists() -> None:
    store = FavoritesStore()
    store.add(_device())
    assert store.remove("spyserver:9.9.9.9:5555")
    assert not store.remove("spyserver:9.9.9.9:5555")  # already gone

    reloaded = FavoritesStore()
    reloaded.load()
    assert reloaded.all() == []


def test_set_note_round_trips() -> None:
    store = FavoritesStore()
    device = _device()
    store.add(device)
    assert store.set_note(device.id, "night dx")
    assert store.get(device.id).note == "night dx"

    reloaded = FavoritesStore()
    reloaded.load()
    assert reloaded.get(device.id).note == "night dx"
