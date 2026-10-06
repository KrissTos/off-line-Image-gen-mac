"""/api/models reports models found in the global HF cache, not ./models."""
from fastapi.testclient import TestClient


def test_models_available_reads_global_cache(monkeypatch, tmp_path):
    import server
    from core import workflow_utils
    app_mod = server._app()
    repo = next(iter(workflow_utils._APP_MODEL_REPOS.values()))
    (tmp_path / ("models--" + repo.replace("/", "--"))).mkdir()
    monkeypatch.setattr(app_mod, "get_local_models_dir", lambda: str(tmp_path))
    data = TestClient(server.app).get("/api/models").json()
    expected = [c for c, r in workflow_utils._APP_MODEL_REPOS.items() if r == repo]
    assert data["available"] == expected
