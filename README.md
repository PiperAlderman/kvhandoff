# kvhandoff

Hand a vLLM prefill to an MLX decoder. vLLM prefills a prompt on NVIDIA GPUs and exports the request's KV cache; [EXO](https://github.com/exo-explore/exo) on Apple Silicon receives it in EXO's disaggregated-prefill wire format and decodes from it. The typical pairing is a DGX Spark (GB10) for prefill, where compute is plentiful, and a Mac Studio for decode, where memory bandwidth and capacity are.

kvhandoff is out-of-tree and has no hard dependencies. It contains:

- **Two vLLM KV connectors** (`kv_producer` only), loaded through vLLM's `vllm.general_plugins` entry point:
  - `KVHandoffPagedConnector` for standard attention (tested with Llama-3.3-70B). It writes each tagged request's K and V per layer as the forward pass reaches it.
  - `KVHandoffDSV4Connector` for DeepSeek V4. It exports each finished request's whole per-layer state (sliding window, compressor and indexer state), converted from vLLM's paged fp8 caches to mlx-lm's `DeepseekV4Cache` layout.
- **An export store** on a tmpfs: fixed slots, per-slot locks, hold and release, immediate back-pressure, and failure records so a reader fails fast instead of timing out.
- **A transfer service** (`kvhandoff-transfer`) that serves exports over HTTP with a bearer token.
- **An EXO bridge** that answers EXO's remote-prefill requests by sending the prompt to vLLM and streaming the export back in EXO's wire format, plus a four-line hook for EXO's runner (`kvhandoff-exo-hook`).

Status: tested on CPU (197 tests with torch, numpy and an EXO checkout; see section 12 for what skips without them) and on hardware (one DGX Spark with a Mac Studio for Llama-3.3-70B; four DGX Sparks at TP4 with three Macs for DeepSeek V4 Flash). It is young code with a small user base; see Limitations.

## Contents

1. Architecture
2. Install
3. Configuration
4. Cache-interchange header
5. Export store
6. Transfer service
7. EXO integration
8. Security model
9. Sizing
10. Measured results
11. Supported versions
12. Tests
13. Limitations
14. Licence

## 1. Architecture

```
 Mac (EXO decode instance)                 Host running the EXO prefill placeholder and the kvhandoff hook
 ------------------------                  ---------------------------------------------------------------
 remote_prefill()  --- EXO wire --->  EXO PrefillServer -> runner.resolve() -> [kvhandoff hook]
   PrefillRequest(token_ids, start_pos)                                          |
   <-- Header, KVChunk*/ArraysState*, Done --                          kvhandoff.exo.bridge.Bridge
                                                                         | 1. POST /v1/completions
                                                                         |    max_tokens 1,
                                                                         |    kv_transfer_params {"kvhandoff_tag": t}
                                                                         | 2. GET /v1/exports/t (poll)
                                                                         | 3. GET /v1/exports/t/layers/i
                                                                         | 4. POST /v1/exports/t/release
                                                                         v
 vLLM host (for TP, the rank 0 host)
 ------------------------------------------------------------------------------------------------
 vLLM (container)                                    kvhandoff-transfer (system uid, bearer token)
   KVHandoffPagedConnector  (save_kv_layer)               | reads only, plus release markers
   KVHandoffDSV4Connector   (get_finished, rank 0)        | sendfile from tmpfs, seq per response
        | gather / convert (torch, on the GPU)            |
        v                                                 v
   ExportStore  ------------------------->  tmpfs owned by the system uid, outside /dev/shm
     slots/s<k>/{L<i>.bin, header.json, lock}     tags/<t> -> ../slots/s<k>     failed/<t>.json
```

A hand-off takes six steps:

1. EXO asks the bridge for a prefill.
2. The bridge sends the prompt to vLLM with a fresh tag.
3. The connector writes the request's cache into a free slot and publishes it under the tag, with a versioned header.
4. The bridge polls the transfer service until the header appears, or until a failure record appears: every 3 ms for the first 2 s (it asks only after vLLM has answered, when the export is tens of milliseconds from published), then every 20 ms, and every 50 ms after 10 s.
5. The bridge fetches the layers (the next layer prefetched in paged mode, all layers in parallel in DeepSeek mode).
6. The bridge streams them in EXO's wire format, then releases the export.

Every failure before the first byte becomes an exception, so EXO answers with an error and the Mac prefills locally. With `stream_layers`, the bridge sends each layer while vLLM is still prefilling (`docs/layerwise.md`).

**Package layers.** Each imports nothing from the layers above it:

- **Core:** header, store, transfer service, auth, config and logs. Stdlib only.
- **Producer:** needs torch, but not vLLM.
- **vLLM connectors:** need vLLM.
- **EXO bridge:** needs numpy, plus EXO's own protocol module, imported lazily.

```
src/kvhandoff/
  interchange.py               ExportHeader, LayerRef, BlobRef, RopeInfo; to_json/from_json; validate()
  tags.py                      tag rules, kv_transfer_params parsing
  config.py                    ConnectorConfig (extra_config), TransferConfig, BridgeConfig (TOML)
  auth.py                      load_token(file), bearer_ok() (constant time)
  logs.py                      JSON-lines logging
  store.py                     ExportStore (writer), ExportReader (readers), CLI kvhandoff-store {init,check,stats}
  transfer/server.py           TransferServer, CLI kvhandoff-transfer --config
  transfer/client.py           TransferClient, LocalSource, VllmClient
  producer/paged.py            RequestTracker, gather(), PagedWriter
  producer/dsv4_convert.py     DeepSeek V4 converter (pure torch)
  producer/dsv4_export.py      SchedulerQueue, FinishedTracker, DSV4Exporter
  producer/maps.py             reused slot file maps
  vllm/paged_connector.py      KVHandoffPagedConnector(KVConnectorBase_V1, SupportsHMA)
  vllm/dsv4_connector.py       KVHandoffDSV4Connector(KVConnectorBase_V1, SupportsHMA)
  vllm/plugin.py               register() for the vllm.general_plugins entry point
  exo/bridge.py                Bridge.resolve(request, wfile), EXO's ResolveHandler signature
  exo/hook.py                  CLI kvhandoff-exo-hook {install,remove,status} --exo-root
typings/vllm/                  stubs for the vLLM names used (type checking without vLLM installed)
deploy/                        sysusers.d, systemd units (tmpfs mount, transfer service per pair), example configs
docs/layerwise.md              layer-by-layer streaming: design, memory ordering, abort handling
```

**Connector notes.**

- Both connectors take `(vllm_config, role, kv_cache_config)`; vLLM's module-path loader refuses classes without the third argument. Both are `SupportsHMA`, so they keep the hybrid KV cache manager on.
- **Paged, scheduler side.** `RequestTracker` follows a request's block ids across chunked-prefill steps and hands it to the workers in the step where its prefill completes (vLLM's `ExampleConnector` stores a prompt in the step that first schedules it, before later chunks are computed).
- **Paged, worker side.** `save_kv_layer` gathers the request's rows and normalises them to `[2, T, H, D]`. It accepts `[T, 2, H, D]` (v0.30.0 FlashInfer) or `[T, H, 2D]` (vLLM main's standardised view) and refuses anything else, such as MLA. Under tensor parallelism all ranks all-gather the heads and rank 0 writes. `wait_for_save` publishes the step's exports.
- **DeepSeek V4.** vLLM never calls `save_kv_layer` for DeepSeek V4 attention layers, so this connector exports from `get_finished()` on TP rank 0, using the block ids that `request_finished_all_groups()` delays freeing.

## 2. Install

Python 3.12 or later. The core is stdlib only, so kvhandoff installs with `--no-deps` into images that bring their own torch and numpy.

```bash
git clone <this repository> kvhandoff && cd kvhandoff
# On the vLLM side (inside the vLLM image; never let pip replace the image's torch):
pip install --no-deps .
# On the EXO side (into EXO's venv, which already has numpy and msgspec):
uv pip install --no-deps .
# Transfer service and CLIs on the host:
python3 -m venv /opt/kvhandoff/venv && /opt/kvhandoff/venv/bin/pip install .
```

**vLLM host.**

1. **System account.** Install `deploy/sysusers.d/kvhandoff.conf` and run `systemd-sysusers`. The example uses uid/gid 980; pick a number free as both uid and gid on every host, because containers run with `--user <uid>:<gid>`.
2. **Store.** Install `deploy/systemd/srv-kvhandoff.mount` (a tmpfs at `/srv/kvhandoff`, `mode=2750`, owned by that uid, mounted `noswap`) and size it as in section 9. Run `kvhandoff-store check --root /srv/kvhandoff/<name> --slots N --slot-bytes B --uid 980 --require-noswap` first: it exits 1 if the location is unsafe, too small, or a tmpfs that can be swapped out. Then run `kvhandoff-store init` as that uid.
   - **`noswap` needs Linux 6.4 or later.** Without it, an idle store is pushed to swap whenever another process needs memory (on a DGX Spark, every new vLLM start-up), and the next export pays to swap it back in, at about 8 to 10 s/GiB measured. The writer logs `store_swappable` once at start when its store could be swapped out.
   - **With `noswap`, every written byte is pinned RAM** until its slot file is removed, so the tmpfs `size=` is the cap on what the store can pin.
   - For a Docker volume instead of a host mount: `docker volume create --driver local --opt type=tmpfs --opt device=tmpfs --opt o=size=12g,uid=980,gid=980,mode=2750,noswap kvhandoff-store`.
3. **Secrets.** Put the pair's bearer token at `/etc/kvhandoff/<pair>.token` (one value of at least 32 characters, 0640 root:kvhandoff) on both hosts, and, if vLLM gets an API key, `/etc/kvhandoff/<pair>.vllm.key` (bare, same mode) plus `/etc/kvhandoff/<pair>.vllm.env` (`VLLM_API_KEY=<key>`, 0600 root:root).
4. **vLLM container.**
   - Pass `--kv-transfer-config` from `deploy/examples/kv-transfer-config-*.json` and mount the store: `-v /srv/kvhandoff/<name>:/srv/kvhandoff/<name>`.
   - Run as the system uid where the image allows it (`--user 980:980 --group-add <group that can read the model cache>`, `-e HOME=/tmp` with `HF_HOME` and `VLLM_CACHE_ROOT` under `/tmp`), otherwise as root.
   - For an API key, add `--env-file /etc/kvhandoff/<pair>.vllm.env`, never `--api-key` on argv.
   - The store does not need `--ipc=host`. For DeepSeek V4, mount the package on every TP rank (jobs are pickled to every worker); only rank 0 writes the store.
5. **Transfer service.** Install `deploy/systemd/kvhandoff-transfer@.service` and run `systemctl enable --now kvhandoff-transfer@<pair>`. systemd hands the service a private copy of the token (`LoadCredential`), and `/etc/kvhandoff/<pair>.transfer.toml` points `token_file` at `/run/credentials/kvhandoff-transfer@<pair>.service/token`. Bind it to the address the bridge uses.

**EXO prefill host.**

1. Make kvhandoff importable by EXO's interpreter and run `kvhandoff-exo-hook install --exo-root <EXO checkout>`.
2. Put `KVHANDOFF_BRIDGE_CONFIG=<bridge.toml>` in EXO's environment (`deploy/examples/bridge-*.toml`).
3. Add EXO's user to group `kvhandoff` and set `allow_group_readable_token = true` in the bridge TOML.
4. On the decode Mac, enable EXO's disaggregated prefill so its decode instance pulls prefills from this host.

**Logs.** The transfer service logs one JSON line per request to the journal; the connectors log JSON lines through the `kvhandoff` logger in vLLM's log; the bridge logs in EXO's log. `jq 'select(.event=="handoff")'` gives the per-hand-off timings.

## 3. Configuration

Every key that names an address or path has no default. Unknown keys are errors. kvhandoff grew out of a prototype, and that prototype's keys are still accepted where noted.

#### `kv_connector_extra_config` (both connectors)

| Key | Default | Meaning |
| --- | --- | --- |
| `store_path` | required | store root (the prototype's `shared_storage_path` is accepted) |
| `slots` | paged: `max_num_seqs + 1`; dsv4: 16 | slots in the store (the prototype's `retain` is accepted) |
| `hold_seconds` | 120 | an unreleased export is kept at least this long |
| `failed_keep` | 64 | failure records kept |
| `model_id` | vLLM served model name | written to the header; the bridge can require it |
| `export_all` | false | export every request, not only tagged ones |
| `out_dtype` | none | paged only: cast K/V (`float16` or `bfloat16`) |
| `rerope` | `ratio0` | dsv4 only: re-rotate ratio-0 layers to MLX's RoPE, or `none` |
| `rope_check_layers` | 3 | dsv4 only: layers whose RoPE distance to MLX is recorded in `extra` |
| `allow_login_uid_store` | false | accept a `/dev/shm` store owned by a login uid (RemoveIPC risk) |
| `timing_log` | false | log one line per export |
| `stream_layers` | false | paged only: announce each layer in the slot's `stream.log` as soon as it is written, so the bridge can send it while vLLM still prefills (`docs/layerwise.md`); the DeepSeek V4 connector refuses it |
| `direct_plane_copy` | true | paged only: copy K and V into the slot file as two contiguous copies instead of one copy into the strided `[2, :T]` view, which PyTorch stages through a fresh host buffer per layer when the source is on the GPU. Same bytes. Measured on a DGX Spark: with it, warm-slot prefill is within about 1 s of vLLM without a connector (8k 10.1 s against 9.0 s; 32k 61.8 s against 60.4 s), against 19.0 s and 102.3 s for the strided path. Set `false` for the strided path |
| `prefault_slot_maps` | true | paged and dsv4 (for dsv4: each layer file range an export writes; a range `presize_tokens` already covered costs nothing): before each layer's copy, populate this process's page-table entries for the rows it is about to receive (`madvise(MADV_POPULATE_WRITE)` after `reserve`, Linux 5.14+), once per map and row range. Only the written rows are populated. A refused `madvise` is logged once and never fails an export; the `exported` log line carries `prefault_seconds`. With a `noswap` store it removes the cost of the first export into rows a fresh vLLM process has not written (measured on a DGX Spark at 32k: copy 49.1 s without it, 0.15 s with it, plus 0.3 s of prefault). It cannot help a store that was swapped out |
| `presize_tokens` | 32768 | dsv4 only: at start-up (`register_kv_caches`, TP rank 0) size, allocate and pre-fault every slot's layer files for prompts up to this many tokens, capped at `max_model_len`; 0 turns it off. V4 Flash at 32768: about 245 MB per slot, 3.9 GB for 16 slots, resident from start-up on a `noswap` store (see section 9). Skipped if over half the store's free bytes; logged as `presized`. The paged connector ignores it |

**Selecting requests.** A request is exported when its `kv_transfer_params` carry `{"kvhandoff_tag": "<tag>"}` (or every request, with `export_all`). Any client of vLLM's API can set that key, so see section 8 on protecting the vLLM endpoint.

#### `[transfer]` (TOML, `kvhandoff-transfer --config`)

| Key | Default | Meaning |
| --- | --- | --- |
| `bind`, `port`, `store_path` | required | |
| `token_file` | required | unless `insecure_no_auth = true` (tests only) |
| `max_concurrent_transfers` | 16 | |
| `max_connections` | 64 | |
| `io_timeout_s` | 30 | per socket read or write |
| `header_timeout_s` | 10 | a request's line and headers (and the idle wait for the next request on a kept-alive connection) must arrive within this, or the connection is closed |
| `hold_seconds` | 120 | reported in /healthz only |
| `allow_group_readable_token` | false | |
| `log_level` | info | |

#### `[bridge]` (TOML, path in `KVHANDOFF_BRIDGE_CONFIG`)

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | required | `paged` or `dsv4` |
| `vllm_url`, `vllm_model` | required | |
| `vllm_api_key_file` | none | vLLM's API key, sent as a bearer (give vLLM the same key with `VLLM_API_KEY` from an env-file, never `--api-key` on argv) |
| `transfer_url` + `transfer_token_file` | one source required | fetch over HTTP |
| `local_store_path` | (the other source) | paged mode only, same host, needs the store's group |
| `timeout_s` | 50 | |
| `max_inflight` | 4 | |
| `queue_wait_s` | 5 | |
| `fetch_threads` | 8 | |
| `result_cache_entries` | 8 | |
| `result_cache_bytes` | 4 GiB | |
| `pad_tokens` | `block_size + 1` | paged: tokens appended to the prompt. Every prompt token is stored, so 0 is enough: measured at 8k, 0 gave the same answers (9/9) and no slower first token. The default keeps the value most hardware runs used |
| `block_size` | 16 | vLLM `--block-size` |
| `wire_dtype` | the export's | paged: `float16` casts a bf16 export for an fp16 MLX model |
| `expect_model_id` | none | refuse exports from another model |
| `stream_layers` | false | paged only: run vLLM's request in the background and send each layer as soon as the connector has written it; Done only after publish. Works against a connector without `stream_layers` (it then sends the published export) |
| `allow_group_readable_token`, `log_level` | | |

The examples in `deploy/examples/` use documentation addresses (192.0.2.0/24) and parse in the tests.


## 4. Cache-interchange header (format 1.0)

Each export has one JSON document. It is stored in the slot as `header.json` and served at `GET /v1/exports/<tag>`.

```json
{"format": "kvhandoff", "format_version": [1, 0], "kind": "paged_kv | dsv4_state",
 "tag": "p-0123456789abcdef", "seq": 1793561234567890, "slot": 3, "created_unix": 1793561234.5,
 "model": {"id": "llama70b", "num_layers": 80},
 "tokens": {"count": 8017, "first_position": 0},
 "rope": {"style": "neox | gptj", "applied_by": "vllm | mlx | mixed", "theta": 500000.0, "scaling": {...},
          "details": {"ratio0_layers": "mlx", "compressed_layers": "vllm (identical to mlx YaRN)", ...}},
 "producer": {"connector": "KVHandoffPagedConnector", "vllm": "0.30.0", "block_size": 16, "req_id": "...",
              "package": "0.1.0"},
 "error": null,
 "layers": [{"index": 0, "name": "model.layers.0.self_attn.attn", "file": "L0.bin", "nbytes": 1234,
             "blobs": [{"role": "k", "dtype": "float16", "shape": [8017, 8, 128], "offset": 0, "nbytes": ...},
                       {"role": "v", ...}],
             "attrs": {"compress_ratio": 4}}],
 "extra": {"export_seconds": 0.21, "rope_check_max_abs": {"0": 1.6}}}
```

**Versioning.** A reader rejects an unknown major version (`UnsupportedFormatVersion`). Within a known major, it ignores unknown fields.

**Validation.** `validate()` runs before publish and after every decode. It checks:

- The blob dtypes are only those EXO's wire carries (`bfloat16`, `float16`, `float32`).
- `nbytes == prod(shape) * itemsize`, and every blob lies inside its layer file.
- File names are plain names.
- `paged_kv` layers hold exactly `k` and `v`, both `[tokens, heads, head_dim]`.
- A `dsv4_state` layer's first blob is the float32 state header.

**Two kinds of export:**

- **`paged_kv`.** Each layer file is `[2, max_tokens, H, D]`, with the K block first and the V block at offset `max_tokens * H * D * itemsize`. The reader slices each blob without converting it.
- **`dsv4_state`.** The blobs are the EXO DeepSeek V4 transport's `ArraysState` for one layer: a float32 state header (its wire version 1), then the window keys and values, then the present compressor and indexer arrays. Their roles are `state_header`, `window_k`, `window_v`, `compressor.buffer_kv` and so on. The header does not change these bytes. The bridge passes them to EXO unchanged.

**RoPE.**

- Llama keys are stored as vLLM rotated them (neox style, which is the same as MLX).
- DeepSeek V4 ratio-0 layers are re-rotated to MLX's plain RoPE by default (`rerope = "ratio0"`). The compressed layers already match MLX's YaRN.
- The header records which was done, so a reader can refuse a mismatch.

## 5. Export store

**Location.** The store must not be somewhere systemd-logind RemoveIPC can reach. RemoveIPC deletes `/dev/shm` and `/dev/mqueue` objects owned by a login uid (uid ≥ UID_MIN in `/etc/login.defs`) when that user's last session ends.

`ExportStore` refuses a `/dev/shm` location when its process runs as such a uid (`UnsafeStoreLocation`), unless `allow_login_uid_store = true` is set. The deployment closes this off twice:

- the store belongs to a system account (`kvhandoff`; the examples use uid/gid 980), or the writer runs as root, which is exempt too;
- it lives on its own tmpfs (`/srv/kvhandoff`, or a tmpfs Docker volume), not on `/dev/shm`.

**Pre-sized slot files (dsv4).** With `presize_tokens`, the connector sizes, allocates (`posix_fallocate`) and pre-faults (`MADV_POPULATE_WRITE`) every slot's layer files at start-up, so no request pays for fresh tmpfs pages or a remap (measured on a DGX Spark: about 0.2 s at 2k into a fresh slot against 0.017 s into a used one). Sizes come from the converter's own arithmetic (`dsv4_convert.layer_capacity`). Pages a previous vLLM process allocated are reused and not counted again. A longer prompt still grows its files, rounded up to a power of two below 64 MiB. Paged slots are not pre-sized: they hold `max_model_len` rows of every layer, and the paged writer already reserves and pre-faults only the rows each export uses.

**Reader header cache.** `ExportReader` keeps each slot's parsed `header.json`, keyed by the file's device, inode, size, mtime and ctime as read through an open descriptor. The writer only replaces `header.json` by rename (and `begin()` unlinks it), so a reused or republished slot never matches an old entry; every `open()` still takes `LOCK_SH` and checks tag and seq. The transfer service opens the export once per layer request, so a DeepSeek V4 hand-off now parses its 43 kB header once instead of 43 times.

**Slots.** There is a fixed number of slots: paged default `max_num_seqs + 1`, DeepSeek default 16. A slot is reused only when all three hold:

- it is empty, consumed (released by the bridge) or older than `hold_seconds` (default 120);
- no reader holds it (flock `LOCK_SH`);
- the writer is not writing it (`LOCK_EX` from `begin()` to `publish()`).

`begin()` never blocks. When no slot qualifies it raises `StoreFull`, and the connector writes `failed/<tag>.json` instead. The bridge sees that record within one poll and fails at once, rather than waiting out its 50 s budget.

**Clean failure.**

- **Exports.** Every export error is a failure record, and the slot is given back. This covers StoreFull, a full tmpfs, an unsupported layout, a short block table and a missing DeepSeek sub-cache.
- **Full tmpfs.** Pages are allocated with `posix_fallocate` before any mapped write. A full tmpfs is then `ENOSPC`, and so a failure record, rather than a SIGBUS in the vLLM worker.
- **Readers.** A reader that finds its tag gone, reused (its `seq` changed) or mid-rewrite gets `ExportNotFound`, which the transfer service answers as 404 and the bridge raises as `HandoffError`.

**Files deleted under the writer.** If the store's files are deleted under a running writer (RemoveIPC, or an operator running `rm`), the next `begin()` sees that the marker or slot directory is gone. It recreates the layout and bumps the slot generation, so the old file maps are dropped. The prototype instead kept writing into deleted files until vLLM restarted.

**One writer only.** The store allows a single writer. `writer.lock` is held for the writer's lifetime, so a second vLLM pointed at the same store fails at start with `StoreBusy`.

**Restarts.** After a restart, published exports stay readable and the seq keeps growing. Stale tag links are dropped.

**Permissions.** Directories are 2750 and files 0640. Readers in the store's group can read but not write; they cannot release, so their slots free after `hold_seconds`.

## 6. Transfer service

| Endpoint | Auth | Answer |
| --- | --- | --- |
| `GET /healthz` | none | 200 with `{status, version, store: {root_ok, slots, published, held, consumed, failed_records, fs_bytes, fs_free_bytes}, inflight, uptime_s}`; 503 if the store marker is missing. No tags. |
| `GET /v1/exports/<tag>` | bearer | 200 header JSON (`X-KVH-Seq`), 200 with `error` set for a failure record, 404 until published |
| `GET /v1/exports/<tag>/layers/<i>` | bearer | the layer's blobs concatenated in header order, via `sendfile`, under the slot's shared lock; `X-KVH-Seq` |
| `POST /v1/exports/<tag>/release?seq=<n>` | bearer | 204, or 404 if gone or reused |
| `GET /v1/exports/<tag>/stream` | bearer | `stream_layers` only: `{published, header}` with the layers completed so far, or the final header; 200 with `error` set for a failed stream; 404 until begun. Logged at debug level (it is polled) |
| `GET /v1/exports/<tag>/layers/<i>` with `X-KVH-Live: 1` | bearer | `stream_layers` only: a completed layer of an export that is still being written, checked against `X-KVH-Seq` |

**Token.** The service reads the token from a file at start. It refuses a world-readable file, and a group-readable one unless allowed. It never takes a token on argv or in the environment. The comparison is constant time.

**Limits.**

- `max_concurrent_transfers` caps `/v1` requests served at once. Beyond it the service answers 503 with `Retry-After: 1`, and the client backs off and retries within its deadline.
- `max_connections` caps open connections. A connection beyond it gets a raw 503 and is closed before a thread starts.
- `io_timeout_s` applies per socket operation, and `header_timeout_s` to the whole request line and headers, so a client trickling bytes cannot hold a connection.

**Logs.** One JSON line per request, with `peer`, `method`, `route`, `status`, `bytes`, `ms` and `tag`. Layer 200s log at debug level. Auth failures log at warning level, without the presented value.

**Lifecycle.** SIGTERM shuts the service down cleanly. It holds no state between requests, so restarting it is safe.

## 7. EXO integration

`kvhandoff-exo-hook install --exo-root <checkout>` inserts four lines at the top of the runner's prefill `resolve`. They are inert unless `KVHANDOFF_BRIDGE_CONFIG` is set.

- `install` keeps a byte-exact backup.
- `remove` strips exactly those lines, so it is safe after EXO was updated, and deletes the backup if the result matches it.

kvhandoff must be importable by EXO's interpreter: put it on `PYTHONPATH`, or `uv pip install --no-deps` it into EXO's venv.

**Wire format.** The bridge uses EXO's own `exo.worker.disaggregated.protocol`, so it follows whatever EXO's structs are.

- **paged:** `Header(num_layers, dtype=wire, start_pos)`, then one `KVChunk` per layer with rows `[start_pos, n)`, then `Done(n - start_pos)`.
- **dsv4:** `Header(dtype="bfloat16")`, then one `ArraysState` per layer with the stored blobs, then `Done(T - start_pos)`. This is the EXO DeepSeek V4 receiver's convention: the whole V4 state is sent whatever `start_pos` is.

`tests/compat/` feeds the packaged blobs to the EXO DeepSeek V4 receiver's `inject_deepseek_v4_cache` (EXO's DeepSeek V4 cache transport, [exo-explore/exo#2545](https://github.com/exo-explore/exo/pull/2545)). Re-serialising them with `deepseek_v4_cache_to_blobs` gives the same bytes in 9 of 9 cases, with T in {5, 300, 1030} and compress ratio in {0, 4, 128}.

**Concurrency.**

- **Prefill limit.** `max_inflight` caps vLLM prefills from one bridge (default 4). Each waits up to `queue_wait_s` for a place, then fails.
- **dsv4 sharing.** The pipeline ranks of one prompt share one prefill through a per-prompt lock. A rank that waited on an attempt that then failed gets the same error.
- **Result cache.** Results are kept LRU, bounded by entries and by bytes, and an entry that is still being produced is never evicted.
- **Fetch threads.** Layer fetches share one pool of `fetch_threads` across all hand-offs.

**Timeouts.**

- `timeout_s` (default 50 s) covers prefill, export and the first layer. It sits under EXO's 60 s client socket timeout.
- **Streaming bridge, buffered connector** (`stream_layers` on the bridge only): the first layer exists only once the whole prefill is published, so `timeout_s` must cover the longest prefill. At 27k tokens on Llama-3.3-70B the first layer came 49 to 55 s after the request, beyond the default; our runs used 300 s.
- **paged:** once streaming has started, each further layer gets its own `timeout_s`. EXO's timeout is per receive, and a 70B 32k stream is about 10.7 GB.
- **dsv4:** everything happens before the header, so the whole budget applies.

**Failures.**

- **Before the first byte.** The bridge raises, EXO sends its error frame, and the Mac prefills locally. Its cache is untouched, because EXO's client buffers the whole stream before ingesting it.
- **Mid-stream (paged only).** The bridge ends EXO's stream with an error frame, then raises, and the Mac falls back the same way. The export is held for the bridge until it releases it, so a mid-stream failure needs a real fault. With `stream_layers` it can also be a failure of vLLM's request or of the export while it is being written (`docs/layerwise.md` section 2.4).

## 8. Security model

kvhandoff is designed for a prefill host and a decode host that share a private link. Its defaults:

- **Bearer token per pair.** Every `/v1` route of the transfer service needs `Authorization: Bearer <token>`, compared in constant time. Tokens are read from files only, never argv or the environment. A world-readable or group-writable token file is refused, and a group-readable one unless `allow_group_readable_token` is set. Logs never contain the presented value. The clients never follow redirects and ignore proxy settings from the environment, so a token only goes to the configured URL.
- **`/healthz` is open.** It answers without a token, for monitoring, with counts and file-system bytes only. An allowlist in `transfer/server.py` keeps tags and paths out, and a test checks it with every place a tag can live populated.
- **Plain HTTP.** The intended deployment puts the transfer service on loopback (vLLM and the bridge on one host) or on an unrouted point-to-point link between two hosts. The bearer token and the 0640 token file keep out other users and processes; an attacker who can sniff or inject on such a link already controls one of the hosts, which TLS would not change. If the traffic crosses a shared network, put TLS (or mTLS) in front of the service.
- **vLLM API key, recommended.** Any client of vLLM's OpenAI API can put `kvhandoff_tag` in `kv_transfer_params` and so make the connector fill store slots (each held for `hold_seconds` unless released; a full store fails exports cleanly and the Mac prefills locally, so this is denial of service, not a data leak, since reading an export needs the transfer token). Run the vLLM launched with the connector with `VLLM_API_KEY` and keep its port off shared networks; the bridge sends that key (`vllm_api_key_file`). The value in a docker env-file is visible to anyone in the `docker` group through `docker inspect`.
- **Store isolation.** The store is a tmpfs owned by a system account, directories 2750 and files 0640: readers in its group can read but not write or release. Tags and file names are validated, symlinks out of the store are refused, and one writer holds a lock for its lifetime.
- **Rotation.** There is no dual-token support. Rotating a pair's token means replacing the file on both hosts, restarting `kvhandoff-transfer@<pair>`, then restarting EXO on the bridge host (the bridge reads its token files once). Hand-offs in flight fail and the Mac prefills locally.

## 9. Sizing

**paged.** Each layer file is pre-sized to `(max_model_len + 64)` rows of K and of V:

`slot_bytes = layers x 2 x (max_model_len + 64) x kv_heads x head_dim x itemsize`

(`kvhandoff.store.paged_slot_bytes`). For Llama-3.3-70B fp16 at `max_model_len` 34000 that is 80 x 2 x 34064 x 8 x 128 x 2 = 11.2 GB per slot.

- **Actual use.** tmpfs allocates only written pages, so a slot actually costs the largest prompt it has held. Its high-water mark stays allocated until the file is removed.
- **Worst case** is `slots x slot_bytes`: 56 GB for 5 slots.
- **Pinned memory.** With `noswap` (section 2) written slot pages stay in RAM until the file is removed, so `size=` caps what the store pins. Size it to the slots you need: one Llama-3.3-70B fp16 slot at 34000 tokens is 10.4 GiB, and a 12 GiB tmpfs held it in our runs.
- **Unified memory.** On GB10 this is the same memory the GPU uses, so size the tmpfs (the `size=` option) below what `--gpu-memory-utilization` and the host's other services leave free.
- **Full store.** When the tmpfs is full, exports fail cleanly (ENOSPC, then a failure record) and the Mac prefills locally.

**dsv4.** Measured at 61 to 66 MB per export at about 7k tokens, and 199 to 209 MB at about 28k. 16 slots of 28k fit in 3.4 GB. `presize_tokens` makes that high-water mark immediate: at the default 32768, 16 V4 Flash slots take about 3.9 GB from start-up. On a `noswap` store those pages are resident, on GB10 in the memory the GPU shares, and `register_kv_caches` runs after vLLM has sized its KV cache, so they come out of what `--gpu-memory-utilization` left for the rest of the host. Lower it (or set 0) where memory is tight.

## 10. Measured results

All figures are first-token times through EXO's API on the Mac, one user, unless stated. Prompt classes 2k, 8k and 32k held about 1.7k, 6.7k and 27k tokens.

**Llama-3.3-70B.** vLLM v0.30.0, NVFP4 weights, fp16 KV, on one DGX Spark; EXO with `mlx-community/Llama-3.3-70B-Instruct-4bit` on one Mac Studio (M3 Ultra, 96 GB); 10 GbE between them; warm store.

| | 2k | 8k | 32k |
| --- | --- | --- | --- |
| Mac Studio alone | 14.0 s | 58.2 s | 265.5 s |
| kvhandoff, buffered | 1.9 to 2.1 s | 10.2 to 11.3 s | 60.0 to 63.1 s |
| kvhandoff, `stream_layers` on the Spark | 1.4 s | 8.5 to 8.7 s | 50.0 s |
| kvhandoff, streaming at both ends (with EXO's per-layer ingest) | 1.4 to 2.0 s | 8.0 s | 48.7 s |

- The 32k streaming rows are one clean warm run each; the other warm run in each queued behind an aborted cold prefill (55.6 s and 86.2 s). The "both ends" row needs an EXO per-layer ingest change that is not merged yet.
- **The buffered 32k path has about 10 s of headroom under EXO's timeout.** EXO's client allows 60 s per receive, the first one included, and the bridge sends nothing until vLLM's export is published. The first byte reached the Mac 48.6 to 51.4 s after the request (warm, cold, and a vLLM process's first export; the whole transfer took 58 to 61 s), so a prefill about 20 % slower would cross it. [exo-explore/exo#2575](https://github.com/exo-explore/exo/pull/2575) (not merged yet) makes that first wait grow with the prompt. Use `stream_layers` on both connector and bridge for long prompts.
- **A streaming bridge in front of a buffered connector** completes at 32k since the bridge's poll fix (`stream_poll_sleep`): 3 of 3, first token 60.1 to 65.6 s, with `timeout_s` 300 (section 7) and the Mac on EXO with the open remote-prefill PRs. Before the fix, the bridge's poll back-off overflowed after about 37 s, so every such hand-off failed and the Mac prefilled locally.
- **Failure drills at 8k** (the Mac on EXO with the open remote-prefill PRs). A full store tmpfs (the connector recorded `StoreFull: store filesystem full`, and vLLM stayed healthy), StoreFull (the only slot held by an unconsumed export) and a link dropped between the Mac and the bridge each ended in a correct answer from a local prefill, and the next request ran with no fallback. The Mac fell back 7.3 to 7.4 s after the request in the two store drills, and about 25 s after it on the dropped link (the TCP keepalive of exo-explore/exo#2575; stock EXO waits for its 60 s receive timeout). A vLLM frozen for 62 s mid-prefill completed remotely with #2575 (first token 71.4 s).
- **Cold store (a fresh vLLM process, store already populated), with `noswap` and `prefault_slot_maps`:** the buffered cold 32k export took 50.0 s against 47.5 s warm, with a first token of 62.1 s and no fallback; cold 8k was 10.9 s buffered and 9.0 s streamed. The streamed cold 32k hand-off completed in 79.9 s, about 30 s slower than warm (not yet explained), and needed an EXO change that reports progress per received layer ([exo-explore/exo#2553](https://github.com/exo-explore/exo/pull/2553), not merged yet) so EXO's 90 s runner-stall check does not fire.
- **Chunked prefill** (`--max-num-batched-tokens 8192`) with streaming at both ends: warm 32k first token 47.6 to 47.8 s.
- Answer quality was 9/9 on a fixed check set in every configuration, with no local fallbacks and no failed exports.
- The connector's own cost, measured straight to vLLM with `max_tokens 1` and `direct_plane_copy` on: 1.36 s against 1.32 s without a connector at 2k, 10.1 s against 9.0 s at 8k, 61.8 s against 60.4 s at 32k.
- A 32k hand-off is about 9 GB of fp16 KV; with streaming, the transfer finishes within 0.1 s of vLLM's prefill, and the prefill itself is not slower.
- Mac decode speed is about the same with the hand-off (15.6 against 16.0 tok/s per user, measured on the prototype). Match the KV dtype to the MLX model (`wire_dtype = "float16"` for a 4-bit MLX Llama): bf16 KV slowed Mac decode by 25 to 34 percent.

**DeepSeek V4 Flash.** vLLM at TP4 on four DGX Sparks, the export served over a point-to-point RoCE link to the bridge host; EXO decoding on three Macs as a pipeline (a Mac Studio M3 Ultra and two Mac minis M5 Pro).

| | 2k | 8k | 32k |
| --- | --- | --- | --- |
| Macs alone | 8.3 to 9.0 s | 16.5 to 16.8 s | 58 to 70 s |
| kvhandoff | 1.97 s | 4.43 s | 15.03 s |

- 4 x 8k at once: 18.1 s; 2 x 32k at once: 30.2 s; no fallbacks. Decode 21.3 tok/s for one user (22 tok/s with the Macs alone).
- Quality 8/9 with and without the hand-off (same check set, generous thinking budget).
- Export cost inside vLLM on rank 0: 61 to 66 MB per request at about 7k tokens, 199 to 209 MB at about 28k, 0.02 to 1.1 s.
- Back-pressure (17 exports into 16 slots gives 16 published and one StoreFull) and the RemoveIPC rule (the store survived the last login session ending) were checked on hardware. The RemoveIPC check also passed on the Llama path, with EXO run as a system service: the login user's `/dev/shm` canary was deleted, and the store and hand-offs carried on.

## 11. Supported versions

| Component | Hardware-tested | Code checked against |
| --- | --- | --- |
| vLLM, paged connector | v0.30.0 (FlashInfer layout `[blocks, 2, bs, H, D]`) | v0.30.0 tag `base.py` and `factory.py` (abstract methods, 3-argument constructor, HMA rule; the test stub enforces the same abstract set); main b8c04dd: the same plus the plugin loader and SchedulerOutput fields |
| vLLM, DeepSeek V4 connector | 0.23.1rc1.dev1053+gf2317c227 | main: `request_finished_all_groups` unchanged |
| EXO | 871caad4 plus the DeepSeek V4 cache transport (and pipeline layer offsets for multi-Mac pipelines), proposed upstream | protocol.py of 871caad4; the DeepSeek V4 receiver of exo-explore/exo#2545 |
| Python | 3.12 (vLLM images), 3.13 (EXO) | tests on 3.12, 3.13 and 3.14 |

**Untested vLLM builds.** A connector on a vLLM version not in `kvhandoff.vllm.TESTED_VLLM` logs `untested_vllm` at start, and keeps running.

**Layout on vLLM main.** vLLM main presents layers as a standardised `[B, H, N, C]` view. kvhandoff handles that view when C = 2 x head_dim, but this has not been run on main.

**Decode side.** Llama-style models need no EXO change. DeepSeek V4 needs EXO's DeepSeek V4 cache transport for disaggregated prefill ([exo-explore/exo#2545](https://github.com/exo-explore/exo/pull/2545), not yet merged), and a multi-Mac pipeline also needs pipeline-rank layer offsets.

## 12. Tests

```bash
uv venv -p 3.12 .venv && VIRTUAL_ENV=.venv uv pip install --index-url https://download.pytorch.org/whl/cpu torch \
  && VIRTUAL_ENV=.venv uv pip install numpy msgspec pytest ruff basedpyright
EXO_SRC=<EXO checkout>/src .venv/bin/python -m pytest      # without EXO_SRC, the EXO end-to-end tests skip
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/basedpyright --pythonpath .venv/bin/python
```

The tests run on CPU, with a vLLM stub (`tests/fakes/`) that enforces the real base class's abstract methods, a fake vLLM HTTP server that exports through the real writers, the real transfer service, and EXO's own protocol module to decode. They cover the header's validation and version policy, token files and config parsing, the store (retention, back-pressure, reuse detection, files deleted under the writer, restart recovery, the RemoveIPC rule), the transfer service (auth, limits, logs that never contain the token), the paged path against both vLLM layouts and under tensor parallelism, the DeepSeek V4 converter (every blob's sha256 against the prototype converter that ran on hardware, for 33 cases), concurrent and failing hand-offs end to end, and layer-by-layer streaming (aborts, slot reuse, out-of-order writers). `tests/compat/` cross-checks the DeepSeek V4 blobs against EXO's receiver under MLX (9 of 9 byte-identical).

**Results** (Python 3.12.12, torch 2.15.0+cpu, fresh venvs):

| Environment | Result |
| --- | --- |
| torch, numpy, msgspec, `EXO_SRC` set to an EXO checkout at 871caad4 | 197 passed |
| the same without `EXO_SRC` | 171 passed, 26 skipped (EXO end-to-end tests) |
| pytest only, no torch | 85 passed, 8 skipped (7 modules that need torch, as whole modules, and 1 EXO test) |

**Short-prompt overhead benchmark** (`python3 bench/short_prompt_overhead_bench.py src`, runs against any tree; x86 CPU, `/dev/shm`, medians). On a CPU a fresh page costs a CPU fault; a GB10's device-to-host copy pays 30 to 60 us per fresh 4 KiB page, so the slot-file gap on a DGX Spark is larger (about 0.2 s at 2k):

| Measure | Before | After |
| --- | --- | --- |
| V4 Flash 1882-token export write, fresh slot | 28.6 to 29.8 ms | 8.6 to 8.9 ms presized (23.2 to 24.2 ms fresh) |
| 43 `ExportReader.open()`, in sequence / 8 threads | 96 / 110 to 113 ms | 5.2 / 17 to 20 ms |
| header wait overshoot after publish, median (max) | 11 to 15 ms (47 to 60) | 3.6 to 3.9 ms (5.6) |

Set `KVHANDOFF_FORBIDDEN_NAMES="name1,name2"` to have the hygiene test also keep your own host and account names out of `src/`.

## 13. Limitations

- **Store on a swappable tmpfs.** Without `noswap` (Linux 6.4+), a store swapped out between vLLM processes makes the next export slow (a cold 32k export took 99.6 s against 47.5 s warm). A buffered 32k hand-off then exceeds EXO's 60 s per-receive timeout and the Mac prefills locally; a streamed one can trip EXO's 90 s runner-stall check, which stock EXO answers with an empty reply.
- **Long buffered hand-offs.** A buffered 32k hand-off of a 70B model gets its first byte to the Mac 48.6 to 51.4 s after the request, about 10 s inside stock EXO's 60 s per-receive timeout (section 10); a slower prefill crosses it. Prefer `stream_layers` at that length.
- **Cold streamed hand-offs** are about 30 s slower than warm at 32k, even with `noswap` and `prefault_slot_maps`. They complete only with an EXO client that reports progress per received layer ([exo-explore/exo#2553](https://github.com/exo-explore/exo/pull/2553), not merged yet).
- **Producer only.** The consumer is EXO. kvhandoff does not hand caches to another vLLM.
- **Narrow model and layout coverage.** The paged connector refuses MLA and other layouts; DeepSeek V4 has its own converter. Each connector was hardware-tested with one vLLM build. The paged connector's tensor-parallel gather is tested on CPU only.
- **DeepSeek V4 numerics.** vLLM's fp8 caches are dequantised to bf16, and ratio-0 layers are re-rotated from vLLM's YaRN to MLX's plain RoPE, so the cache is not bit-exact against MLX prefilling the same prompt (quality matched).
- **EXO coupling.** The hook inserts four lines into EXO's runner, and the bridge needs a placeholder EXO prefill instance on the host that carries it. Both track EXO internals.
- **Linux only** on the producer side (`flock`, `posix_fallocate`, `madvise`, tmpfs).
- **No TLS and no dual-token rotation** (section 8). Without a vLLM API key, any vLLM client can tag requests and fill the store's slots (section 8).
- **Layer-by-layer streaming is off by default.** It has been measured warm at three prompt sizes, cold, and with chunked prefill; failure drills with streaming at both ends and under concurrent load are still to do.

## 14. Licence

Copyright 2026 The kvhandoff Authors. Licensed under the Apache License, Version 2.0: see `LICENSE` and `NOTICE`.
