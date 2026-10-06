import asyncio


def test_run_events_logs_traceback(monkeypatch, tmp_path, capsys):
    import server
    from core import run_store
    run = run_store.create_run(tmp_path, {"prompt": "p", "seed": 1}, [])

    class Boom:
        async def generate(self, params):
            raise RuntimeError("kaboom-xyz")
            yield  # pragma: no cover
    monkeypatch.setattr(server, "_mgr", lambda: Boom())
    monkeypatch.setattr(run_store, "trash", lambda p: None)

    async def go():
        async for _ in server._run_events(run, {}, 1):
            pass
    try:
        asyncio.run(go())
    except RuntimeError:
        pass
    assert "Traceback" in capsys.readouterr().err
