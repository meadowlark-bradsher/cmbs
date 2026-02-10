from cmbs.belief_server import BeliefServer, OntologyBundle
from cmbs.stores import InMemoryStore


def test_belief_server_store_recovery_round_trip():
    store = InMemoryStore()
    server = BeliefServer(store=store)
    session_id, before = server.declare_session(
        ontology=OntologyBundle(
            hypothesis_space_id="test",
            hypothesis_version="v1",
            causal_graph_ref="graph",
            causal_graph_version="v0",
        ),
        hypotheses=["H1", "H2", "H3"],
    )
    applied, ignored, after, event_id = server.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-1",
        eliminated=["H1", "H3"],
    )
    assert applied == ["H1", "H3"]
    assert ignored == []
    assert set(store.get_survivors(session_id)) == set(after.survivors)

    audit = server.audit_trace(session_id)
    recovered = store.recover(session_id)

    restarted = BeliefServer(store=store)
    restarted.restore_session(session_id=session_id, recovered=recovered, audit_events=audit)
    restored = restarted.query_belief(session_id)
    assert set(restored.survivors) == set(after.survivors)

    replay_applied, replay_ignored, replay_snapshot, replay_event_id = restarted.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-1",
        eliminated=["H1", "H3"],
    )
    assert replay_event_id == event_id
    assert replay_applied == applied
    assert replay_ignored == ignored
    assert set(replay_snapshot.survivors) == set(after.survivors)
