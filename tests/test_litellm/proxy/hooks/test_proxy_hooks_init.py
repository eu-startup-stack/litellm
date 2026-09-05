"""Layering guard: litellm.llms.* must not transitively import
litellm.proxy.utils, which would reintroduce an import cycle.
"""

import importlib
import sys

import pytest


def test_isolation_module_does_not_pull_in_proxy_utils():
    """litellm.llms.base_llm.managed_resources.isolation must be importable
    in isolation without pulling in litellm.proxy.* — otherwise
    litellm.proxy.hooks consumers would re-enter the proxy namespace mid-init.
    """
    for mod in [
        "litellm.proxy.utils",
        "litellm.proxy.management_endpoints.common_utils",
        "litellm.llms.base_llm.managed_resources.isolation",
    ]:
        sys.modules.pop(mod, None)

    importlib.import_module("litellm.llms.base_llm.managed_resources.isolation")
    assert "litellm.proxy.utils" not in sys.modules
    assert "litellm.proxy.management_endpoints.common_utils" not in sys.modules

