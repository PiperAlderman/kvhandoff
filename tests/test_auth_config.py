from __future__ import annotations

from pathlib import Path

import pytest

from conftest import write_token
from kvhandoff.auth import TokenFileError, bearer_ok, load_token
from kvhandoff.config import BridgeConfig, ConfigError, ConnectorConfig, TransferConfig, load_toml


# ----------------------------------------------------------------------------------------------- auth
def test_token_file_mode_600_is_read(tmp_path: Path) -> None:
    p = write_token(tmp_path / "tok", "x" * 40)
    assert load_token(p) == "x" * 40


@pytest.mark.parametrize("mode", [0o644, 0o604, 0o606])
def test_world_readable_token_is_refused(tmp_path: Path, mode: int) -> None:
    p = write_token(tmp_path / "tok", mode=mode)
    with pytest.raises(TokenFileError, match="other users"):
        load_token(p)


@pytest.mark.parametrize("mode", [0o620, 0o660, 0o670])
def test_group_writable_token_is_refused_even_when_group_read_is_allowed(tmp_path: Path, mode: int) -> None:
    p = write_token(tmp_path / "tok", mode=mode)
    with pytest.raises(TokenFileError, match="group-writable"):
        load_token(p, allow_group_read=True)


def test_group_readable_token_needs_explicit_permission(tmp_path: Path) -> None:
    p = write_token(tmp_path / "tok", mode=0o640)
    with pytest.raises(TokenFileError, match="group-readable"):
        load_token(p)
    assert load_token(p, allow_group_read=True)


def test_short_or_missing_token_is_refused_without_echoing_it(tmp_path: Path) -> None:
    p = write_token(tmp_path / "tok", "short-secret")
    with pytest.raises(TokenFileError) as e:
        load_token(p)
    assert "short-secret" not in str(e.value)
    with pytest.raises(TokenFileError, match="No such file"):
        load_token(tmp_path / "missing")
    (tmp_path / "dir").mkdir()
    with pytest.raises(TokenFileError, match="regular file"):
        load_token(tmp_path / "dir")


def test_bearer_check() -> None:
    tok = "a" * 40
    assert bearer_ok(f"Bearer {tok}", tok)
    assert bearer_ok(f"bearer {tok}", tok)
    assert not bearer_ok(None, tok)
    assert not bearer_ok("", tok)
    assert not bearer_ok(f"Basic {tok}", tok)
    assert not bearer_ok("Bearer " + "a" * 39, tok)


# ----------------------------------------------------------------------------------------------- config
def test_connector_config_needs_a_store_path_and_has_no_path_default() -> None:
    with pytest.raises(ConfigError, match="store_path"):
        ConnectorConfig.from_extra({})
    c = ConnectorConfig.from_extra({"store_path": "/srv/x"})
    assert c.store_path == "/srv/x" and c.slots is None and c.rerope == "ratio0" and not c.allow_login_uid_store


def test_connector_config_maps_the_prototype_keys() -> None:
    c = ConnectorConfig.from_extra({"shared_storage_path": "/s", "retain": 5})
    assert (c.store_path, c.slots) == ("/s", 5)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"store_path": "/s", "typo_key": 1}, "unknown key"),
        ({"store_path": "/s", "slots": 0}, "slots"),
        ({"store_path": "/s", "out_dtype": "float32"}, "out_dtype"),
        ({"store_path": "/s", "rerope": "yes"}, "rerope"),
        ({"store_path": "/s", "export_all": "true"}, "export_all"),
    ],
)
def test_connector_config_errors(extra: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        ConnectorConfig.from_extra(extra)


def test_transfer_config_requires_token_unless_explicitly_insecure() -> None:
    base = {"bind": "127.0.0.1", "port": 9000, "store_path": "/s"}
    with pytest.raises(ConfigError, match="token_file"):
        TransferConfig.from_mapping(base)
    assert TransferConfig.from_mapping({**base, "insecure_no_auth": True}).token_file is None
    c = TransferConfig.from_mapping({**base, "token_file": "/etc/k/t"})
    assert c.max_concurrent_transfers == 16 and c.io_timeout_s == 30.0
    with pytest.raises(ConfigError, match="bind"):
        TransferConfig.from_mapping({"port": 1, "store_path": "/s", "token_file": "/t"})
    with pytest.raises(ConfigError, match="port"):
        TransferConfig.from_mapping({**base, "token_file": "/t", "port": 70000})


def test_bridge_config_source_rules() -> None:
    base = {"mode": "paged", "vllm_url": "http://vllm:8000", "vllm_model": "m"}
    with pytest.raises(ConfigError, match="exactly one"):
        BridgeConfig.from_mapping(base)
    with pytest.raises(ConfigError, match="exactly one"):
        BridgeConfig.from_mapping({**base, "local_store_path": "/s", "transfer_url": "http://t:1", "transfer_token_file": "/t"})
    with pytest.raises(ConfigError, match="transfer_token_file"):
        BridgeConfig.from_mapping({**base, "transfer_url": "http://t:1"})
    with pytest.raises(ConfigError, match="dsv4 mode"):
        BridgeConfig.from_mapping({**base, "mode": "dsv4", "local_store_path": "/s"})
    with pytest.raises(ConfigError, match="http"):
        BridgeConfig.from_mapping({**base, "vllm_url": "file:///etc/passwd", "local_store_path": "/s"})
    c = BridgeConfig.from_mapping({**base, "local_store_path": "/s"})
    assert c.effective_pad_tokens == 17 and c.timeout_s == 50.0
    assert BridgeConfig.from_mapping({**base, "local_store_path": "/s", "pad_tokens": 0}).effective_pad_tokens == 0


def test_toml_loading(tmp_path: Path) -> None:
    p = tmp_path / "t.toml"
    p.write_text('[transfer]\nbind = "127.0.0.1"\nport = 9\nstore_path = "/s"\ntoken_file = "/t"\n')
    assert TransferConfig.from_mapping(load_toml(str(p), "transfer")).port == 9
    with pytest.raises(ConfigError, match="no \\[bridge\\]"):
        load_toml(str(p), "bridge")


def test_shipped_example_configs_parse() -> None:
    import json

    ex = Path(__file__).resolve().parent.parent / "deploy" / "examples"
    assert TransferConfig.from_mapping(load_toml(str(ex / "transfer.toml"), "transfer")).port == 8093
    assert BridgeConfig.from_mapping(load_toml(str(ex / "bridge-dsv4.toml"), "bridge")).mode == "dsv4"
    assert BridgeConfig.from_mapping(load_toml(str(ex / "bridge-llama.toml"), "bridge")).wire_dtype == "float16"
    for name in ("kv-transfer-config-llama.json", "kv-transfer-config-dsv4.json"):
        doc = json.loads((ex / name).read_text())
        assert ConnectorConfig.from_extra(doc["kv_connector_extra_config"]).store_path.startswith("/srv/kvhandoff/")
