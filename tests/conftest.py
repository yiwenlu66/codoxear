import pytest


@pytest.fixture(autouse=True, scope="session")
def _init_server_runtime():
    """Initialize server runtime state once for the test session.

    server.py defers HMAC_SECRET, MANAGER, and Handler creation into
    _init_runtime() so that importing server.py alone does not trigger
    filesystem mutation or thread creation. Tests that need these values
    call _init_runtime() explicitly via this session-scoped fixture.
    """
    from codoxear import server
    server._init_runtime()
    yield
