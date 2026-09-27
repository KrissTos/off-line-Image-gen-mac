"""--no-auto-shutdown must reach the module uvicorn actually serves.

Regression: `uvicorn.run("server:app")` imports `server` as a fresh module, so setting
the global in `__main__` never disabled the heartbeat watcher — the server shut down
60 s after start with no browser attached. The flag now travels via an env var.
"""
import importlib


def test_env_var_disables_auto_shutdown(monkeypatch):
    import server
    monkeypatch.setenv("IMAGEGEN_NO_AUTO_SHUTDOWN", "1")
    assert importlib.reload(server)._auto_shutdown is False


def test_auto_shutdown_default_on(monkeypatch):
    import server
    monkeypatch.delenv("IMAGEGEN_NO_AUTO_SHUTDOWN", raising=False)
    assert importlib.reload(server)._auto_shutdown is True
