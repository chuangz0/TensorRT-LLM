# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV transfer config file: YAML -> ``KVTransferConfig`` validation, defaults and refusals.
Import-light like the ``kv_transfer`` suite: nothing here needs ``tensorrt_llm``.
"""

import textwrap

import pytest

__extra_import_path__ = ["~/tensorrt_llm/_torch"]
from disaggregation.backends.config import (  # noqa: E402
    BACKEND_ROLES,
    KV_TRANSFER_CONFIG_ENV,
    KVTransferConfig,
    load_kv_transfer_config,
)

pytestmark = pytest.mark.cpu_only

EXAMPLE_E2E_YAML = textwrap.dedent(
    """
    fetch_timeout_s: 30
    publish_timeout_s: 60
    probe_timeout_s: 0.05
    backends:
      - name: shared-store
        type: mooncake
        roles: [fetch, publish]
        master_server_address: 127.0.0.1:50051
        protocol: tcp
        local_hostname: 127.0.0.1
        global_segment_size: 0
        landing: host
        namespace: tinyllama-e2e
    """
)


def write_yaml(tmp_path, text: str) -> str:
    path = tmp_path / "kv_transfer.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return str(path)


def load(tmp_path, text: str) -> KVTransferConfig:
    return load_kv_transfer_config(write_yaml(tmp_path, text))


def test_env_var_name_is_the_documented_one():
    assert KV_TRANSFER_CONFIG_ENV == "TRTLLM_KV_TRANSFER_CONFIG"
    assert BACKEND_ROLES == ("fetch", "publish")


def test_the_documented_e2e_example_loads(tmp_path):
    config = load(tmp_path, EXAMPLE_E2E_YAML)
    assert config.fetch_timeout_s == 30
    assert config.publish_timeout_s == 60
    assert config.probe_timeout_s == 0.05
    assert config.close_timeout_s == 30.0  # default
    assert len(config.backends) == 1
    entry = config.backends[0]
    assert entry.name == "shared-store"
    assert entry.type == "mooncake"
    assert entry.hint_key is None
    assert entry.roles == frozenset({"fetch", "publish"})
    assert entry.has_fetch_role and entry.has_publish_role
    # Type-specific keys pass through unread; entry keys do not.
    assert entry.options == {
        "master_server_address": "127.0.0.1:50051",
        "protocol": "tcp",
        "local_hostname": "127.0.0.1",
        "global_segment_size": 0,
        "landing": "host",
        "namespace": "tinyllama-e2e",
    }
    assert not set(entry.options) & {"name", "type", "hint_key", "roles"}


def test_minimal_entry_defaults_to_both_roles_and_finite_timeouts(tmp_path):
    config = load(
        tmp_path,
        """
        backends:
          - name: a
            type: fake
        """,
    )
    # Every wait on a peer or a store is bounded by default; ``null`` must be asked for.
    assert config.fetch_timeout_s == 30.0 and config.publish_timeout_s == 60.0
    assert config.unlaunched_timeout_s == 30.0
    assert config.fetch_wait_timeout_s == 30.0
    assert config.probe_timeout_s == 1.0
    assert config.backends[0].roles == frozenset(BACKEND_ROLES)
    assert config.backends[0].options == {}


def test_publish_only_role(tmp_path):
    config = load(
        tmp_path,
        """
        backends:
          - name: a
            type: fake
            roles: [publish]
        """,
    )
    entry = config.backends[0]
    assert entry.has_publish_role and not entry.has_fetch_role


def test_backend_order_is_kept(tmp_path):
    config = load(
        tmp_path,
        """
        backends:
          - {name: first, type: fake}
          - {name: second, type: fake, hint_key: ctx}
        """,
    )
    assert [e.name for e in config.backends] == ["first", "second"]
    assert config.backends[1].hint_key == "ctx"


def test_unknown_top_level_key_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown kv transfer config keys.*probe_timeout"):
        load(
            tmp_path,
            """
            probe_timeout: 1
            backends:
              - {name: a, type: fake}
            """,
        )


@pytest.mark.parametrize("old_key", ["landing_wait_timeout_s"])
def test_retired_top_level_key_is_refused_by_name(tmp_path, old_key):
    with pytest.raises(ValueError, match=f"unknown kv transfer config keys.*{old_key}"):
        load(
            tmp_path,
            f"""
            {old_key}: 30
            backends:
              - {{name: a, type: fake}}
            """,
        )


@pytest.mark.parametrize("roles", ["[fetch, foo]", "[]", "[publish, publish, nope]"])
def test_bad_roles_are_refused(tmp_path, roles):
    with pytest.raises(ValueError, match="roles must be a non-empty subset"):
        load(
            tmp_path,
            f"""
            backends:
              - name: a
                type: fake
                roles: {roles}
            """,
        )


@pytest.mark.parametrize(
    "entry, missing",
    [("- {name: a}", "type"), ("- {type: fake}", "name"), ("- {name: '', type: fake}", "name")],
)
def test_entry_without_name_or_type_is_refused(tmp_path, entry, missing):
    with pytest.raises(ValueError, match=f"non-empty '{missing}'"):
        load(tmp_path, f"backends:\n  {entry}\n")


@pytest.mark.parametrize("text", ["fetch_timeout_s: 1\n", "backends: {a: 1}\n", "backends: 3\n"])
def test_backends_must_be_a_list(tmp_path, text):
    with pytest.raises(ValueError, match="needs a 'backends' list"):
        load(tmp_path, text)


def test_empty_backends_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one backend"):
        load(tmp_path, "backends: []\n")


@pytest.mark.parametrize("text", ["- a\n- b\n", "just a string\n", ""])
def test_non_mapping_file_is_refused(tmp_path, text):
    with pytest.raises(ValueError, match="expected a mapping"):
        load(tmp_path, text)


def test_duplicate_backend_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        load(
            tmp_path,
            """
            backends:
              - {name: a, type: fake}
              - {name: a, type: other}
            """,
        )


@pytest.mark.parametrize(
    "line, message",
    [
        ("fetch_timeout_s: 0", "fetch_timeout_s must be > 0 or null"),
        ("publish_timeout_s: -1", "publish_timeout_s must be > 0 or null"),
        ("unlaunched_timeout_s: 0", "unlaunched_timeout_s must be > 0 or null"),
        ("fetch_wait_timeout_s: 0", "fetch_wait_timeout_s must be > 0 or null"),
        ("probe_timeout_s: -0.1", "probe_timeout_s must be >= 0"),
        ("close_timeout_s: 0", "close_timeout_s must be > 0"),
    ],
)
def test_limits_out_of_range_are_refused(tmp_path, line, message):
    with pytest.raises(ValueError, match=message):
        load(tmp_path, f"{line}\nbackends:\n  - {{name: a, type: fake}}\n")


def test_null_timeouts_and_zero_probe_are_allowed(tmp_path):
    config = load(
        tmp_path,
        """
        fetch_timeout_s: null
        publish_timeout_s: null
        unlaunched_timeout_s: null
        fetch_wait_timeout_s: null
        probe_timeout_s: 0
        backends:
          - {name: a, type: fake}
        """,
    )
    assert config.fetch_timeout_s is None and config.probe_timeout_s == 0
    assert config.unlaunched_timeout_s is None and config.fetch_wait_timeout_s is None


def test_missing_file_raises_os_error(tmp_path):
    with pytest.raises(OSError):
        load_kv_transfer_config(str(tmp_path / "nope.yaml"))
