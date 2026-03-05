# CMBS IR Snapshot: Repo Map

## Scope
This map focuses on active code paths. `archive/` is present but treated as historical/deprecated material (`docs-site/REPOSITORY_STRUCTURE.md:160`).

## Tree (depth ~4)
```text
.
├── cmbs/
│   ├── __init__.py
│   ├── core.py
│   ├── belief_server.py
│   ├── belief_api.py
│   ├── belief_state.py
│   ├── spi/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── elimination_store.py
│   │   └── hypothesis_provider.py
│   ├── stores/
│   │   ├── __init__.py
│   │   └── memory.py
│   └── adapters/
│       ├── __init__.py
│       ├── types.py
│       ├── itbench/
│       │   ├── __init__.py
│       │   ├── adapter.py
│       │   ├── kit.py
│       │   ├── oracle.py
│       │   └── kits/
│       ├── twenty_questions/
│       │   ├── __init__.py
│       │   ├── adapter.py
│       │   ├── kit.py
│       │   ├── oracle.py
│       │   └── kits/
│       └── legacy/
│           ├── __init__.py
│           └── replay.py
├── tests/
│   ├── conftest.py
│   ├── test_v0_core.py
│   ├── test_invariants.py
│   ├── test_belief_server_spi_smoke.py
│   ├── test_belief_server_store_integration.py
│   ├── test_elimination_store.py
│   ├── test_spi_belief_state.py
│   └── test_legacy_adapter.py
├── examples/
│   ├── run_20q.py
│   └── run_itbench.py
├── docs-vault/
│   ├── BELIEF_SERVER_SPEC.md
│   ├── ELIMINATION-STORE-SPI.md
│   ├── ARCHITECTURE.md
│   └── ...
├── docs-site/
│   └── REPOSITORY_STRUCTURE.md
├── Dockerfile
├── README.md
├── requirements.txt
└── pytest.ini
```

## Entrypoints
- HTTP app entrypoint: `cmbs/belief_api.py:13` (`app = FastAPI(...)`) with routes starting at `declare_session` (`cmbs/belief_api.py:80`).
- Container runtime entrypoint: `Dockerfile:12` runs `uvicorn cmbs.belief_api:app`.
- Example script entrypoints:
  - `examples/run_20q.py:5` (`main`) and `examples/run_20q.py:43` (`if __name__ == "__main__":`).
  - `examples/run_itbench.py:5` (`main`) and `examples/run_itbench.py:49` (`if __name__ == "__main__":`).
- Library/public API entrypoint: `cmbs/__init__.py:38` (`__all__` export surface).

## Tests
- Pytest root config: `pytest.ini:1` and `pytest.ini:2` (`testpaths = tests`).
- Import-path bootstrap fixture: `tests/conftest.py:6` (`REPO_ROOT` injection into `sys.path`).
- Core/invariant coverage: `tests/test_v0_core.py`, `tests/test_invariants.py`.
- Server + SPI/store coverage:
  - `tests/test_belief_server_spi_smoke.py`
  - `tests/test_belief_server_store_integration.py`
  - `tests/test_elimination_store.py`
- SPI belief-state coverage: `tests/test_spi_belief_state.py`.
- Legacy replay adapter coverage: `tests/test_legacy_adapter.py`.
