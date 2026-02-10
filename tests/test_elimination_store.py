from cmbs.spi.elimination_store import EliminationProvenance
from cmbs.stores import InMemoryStore


def _provenance() -> EliminationProvenance:
    return EliminationProvenance(source_id="test", trigger="unit")


def test_store_create_session_get_survivors():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2"}))
    assert store.get_survivors("s1") == frozenset({"H1", "H2"})


def test_store_eliminate_reduces_survivors():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2", "H3"}))
    result = store.eliminate("s1", {"H1"}, _provenance())
    assert result.applied == frozenset({"H1"})
    assert store.get_survivors("s1") == frozenset({"H2", "H3"})


def test_store_eliminate_idempotent():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2"}))
    store.eliminate("s1", {"H1"}, _provenance())
    result = store.eliminate("s1", {"H1"}, _provenance())
    assert result.applied == frozenset()
    assert result.already_eliminated == frozenset({"H1"})


def test_store_eliminate_unknown_no_effect():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2"}))
    result = store.eliminate("s1", {"H99"}, _provenance())
    assert result.applied == frozenset()
    assert store.get_survivors("s1") == frozenset({"H1", "H2"})
    assert store.get_eliminated("s1") == frozenset()


def test_store_get_eliminated_cumulative():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2", "H3"}))
    store.eliminate("s1", {"H1"}, _provenance())
    store.eliminate("s1", {"H2"}, _provenance())
    assert store.get_eliminated("s1") == frozenset({"H1", "H2"})


def test_store_survivor_and_eliminated_sets_consistent():
    store = InMemoryStore()
    universe = frozenset({"H1", "H2", "H3"})
    store.create_session("s1", universe)
    store.eliminate("s1", {"H1"}, _provenance())
    survivors = store.get_survivors("s1")
    eliminated = store.get_eliminated("s1")
    assert survivors | eliminated <= universe
    assert survivors.isdisjoint(eliminated)


def test_store_recover_consistent():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2", "H3"}))
    store.eliminate("s1", {"H2"}, _provenance())
    recovered = store.recover("s1")
    assert recovered.survivors == store.get_survivors("s1")
    assert recovered.eliminated == store.get_eliminated("s1")
    assert recovered.hypothesis_ids == frozenset({"H1", "H2", "H3"})


def test_store_eliminate_empty_noop():
    store = InMemoryStore()
    store.create_session("s1", frozenset({"H1", "H2"}))
    result = store.eliminate("s1", set(), _provenance())
    assert result.applied == frozenset()
    assert result.already_eliminated == frozenset()
    assert store.get_survivors("s1") == frozenset({"H1", "H2"})
