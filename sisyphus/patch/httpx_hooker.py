import httpx

timeout = httpx.Timeout(10.0, connect=60.0, pool=30.0, read=30.0)
limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)

achat_httpx_client = httpx.AsyncClient(timeout=timeout, limits=limits, verify=False)
aembed_httpx_client = httpx.AsyncClient(timeout=timeout, limits=limits, verify=False)
