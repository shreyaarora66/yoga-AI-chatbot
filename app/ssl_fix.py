"""Configure SSL for machines with missing or intercepted CA certificates."""

from __future__ import annotations

import os
import warnings


def _should_disable_verify() -> bool:
    return os.getenv("SSL_VERIFY", "true").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _patch_requests() -> None:
    import requests

    if getattr(requests.Session, "_yoga_ssl_patched", False):
        return

    original_request = requests.Session.request

    def patched_request(self, method, url, **kwargs):
        kwargs.setdefault("verify", False)
        return original_request(self, method, url, **kwargs)

    requests.Session.request = patched_request
    requests.Session._yoga_ssl_patched = True


def _patch_httpx() -> None:
    import httpx

    for client_cls in (httpx.Client, httpx.AsyncClient):
        if getattr(client_cls, "_yoga_ssl_patched", False):
            continue

        original_init = client_cls.__init__

        def patched_init(self, *args, __orig=original_init, **kwargs):
            kwargs["verify"] = False
            __orig(self, *args, **kwargs)

        client_cls.__init__ = patched_init
        client_cls._yoga_ssl_patched = True


def configure_ssl() -> None:
    try:
        import certifi

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)
    except ImportError:
        pass

    if not _should_disable_verify():
        return

    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    _patch_requests()
    _patch_httpx()
