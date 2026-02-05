from cmbs.belief_server import BeliefServer, OntologyBundle


def _new_server() -> BeliefServer:
    return BeliefServer()


def _declare(server: BeliefServer, hypotheses=None):
    if hypotheses is None:
        hypotheses = ["H1", "H2", "H3", "H4"]
    session_id, snapshot = server.declare_session(
        ontology=OntologyBundle(
            hypothesis_space_id="test",
            hypothesis_version="v1",
            causal_graph_ref="none",
            causal_graph_version="v0",
        ),
        hypotheses=hypotheses,
    )
    return session_id, snapshot


def test_monotonicity():
    server = _new_server()
    session_id, snapshot = _declare(server, ["H1", "H2", "H3"])
    before = set(snapshot.survivors)

    _, _, snapshot, _ = server.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-1",
        eliminated=["H1"],
        justification={"note": "eliminate H1"},
    )
    after = set(snapshot.survivors)

    assert after.issubset(before)


def test_idempotence_duplicate_observation():
    server = _new_server()
    session_id, snapshot = _declare(server, ["H1", "H2", "H3"])

    applied1, ignored1, snapshot1, event1 = server.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-1",
        eliminated=["H1", "H4"],
        justification={"note": "first"},
    )

    applied2, ignored2, snapshot2, event2 = server.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-1",
        eliminated=["H1", "H4"],
        justification={"note": "second"},
    )

    assert applied1 == applied2
    assert ignored1 == ignored2
    assert snapshot1.survivors == snapshot2.survivors
    assert event1 == event2
    assert len(server.audit_trace(session_id)) == 2


def test_order_independence():
    elim_a = ["H1", "H2"]
    elim_b = ["H2", "H3"]

    server1 = _new_server()
    s1, _ = _declare(server1)
    server1.eliminate(
        session_id=s1,
        source_id="adapter://test",
        observation_id="a",
        eliminated=elim_a,
        justification={},
    )
    _, _, snap1, _ = server1.eliminate(
        session_id=s1,
        source_id="adapter://test",
        observation_id="b",
        eliminated=elim_b,
        justification={},
    )

    server2 = _new_server()
    s2, _ = _declare(server2)
    server2.eliminate(
        session_id=s2,
        source_id="adapter://test",
        observation_id="b",
        eliminated=elim_b,
        justification={},
    )
    _, _, snap2, _ = server2.eliminate(
        session_id=s2,
        source_id="adapter://test",
        observation_id="a",
        eliminated=elim_a,
        justification={},
    )

    assert set(snap1.survivors) == set(snap2.survivors)


def test_replayability():
    server = _new_server()
    session_id, snapshot = _declare(server)

    server.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-1",
        eliminated=["H1"],
        justification={"note": "first"},
    )
    _, _, final_snapshot, _ = server.eliminate(
        session_id=session_id,
        source_id="adapter://test",
        observation_id="obs-2",
        eliminated=["H2"],
        justification={"note": "second"},
    )

    events = server.audit_trace(session_id)

    replay = _new_server()
    replay_session_id, _ = _declare(replay)
    for event in events:
        if event.verb != "ELIMINATE":
            continue
        payload = event.payload
        replay.eliminate(
            session_id=replay_session_id,
            source_id=payload["source_id"],
            observation_id=payload["observation_id"],
            eliminated=payload["eliminated"],
            justification=payload.get("justification", {}),
        )

    replay_snapshot = replay.query_belief(replay_session_id)
    assert set(replay_snapshot.survivors) == set(final_snapshot.survivors)
