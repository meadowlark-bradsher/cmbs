# CMBS IR Snapshot: Runtime Capabilities

## Runtime Shape
- Library: Yes. Package exports are defined in `cmbs/__init__.py:9` and `cmbs/__init__.py:38` (`__all__`).
- Server: Yes (HTTP). FastAPI app is declared in `cmbs/belief_api.py:13` and instantiated server state in `cmbs/belief_api.py:14`.
- Not observed in active code: gRPC or raw socket server surfaces (targeted `rg` shows FastAPI/uvicorn only).

## Evidence (File Paths + Symbols)
- HTTP transport module: `cmbs/belief_api.py:1` docstring says "FastAPI transport".
- HTTP route symbols:
  - `declare_session` at `cmbs/belief_api.py:81`
  - `eliminate` at `cmbs/belief_api.py:97`
  - `query_belief` at `cmbs/belief_api.py:114`
  - `audit_trace` at `cmbs/belief_api.py:120`
- Server kernel/session layer: `class BeliefServer` in `cmbs/belief_server.py:129`.
- Core library kernel: `class CMBSCore` in `cmbs/core.py:42`.

## How To Run Locally (from checked-in config)
1. Install deps:
```bash
pip install -r requirements.txt
```
Evidence: runtime deps in `requirements.txt:1`-`requirements.txt:4` (`fastapi`, `pydantic`, `uvicorn`, `pyyaml`).

2. Start HTTP server:
```bash
uvicorn cmbs.belief_api:app --host 0.0.0.0 --port 8000
```
Evidence: same command in container entrypoint `Dockerfile:12`.

3. Run examples:
```bash
python examples/run_20q.py
python examples/run_itbench.py
```
Evidence: script entry guards at `examples/run_20q.py:43` and `examples/run_itbench.py:49`.

4. Run tests:
```bash
pytest
```
Evidence: `pytest.ini:2` points tests to `tests/`.

## Docker Presence
- Dockerfile exists at repo root (`Dockerfile:1`).
- Base image: `python:3.11-slim` (`Dockerfile:1`).
- Container exposes port `8000` (`Dockerfile:10`).
- Container command launches uvicorn (`Dockerfile:12`).
