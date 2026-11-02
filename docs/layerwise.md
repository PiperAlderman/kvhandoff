# Layer-by-layer streaming

`stream_layers` (connector and bridge, off by default) sends each layer's cache to the decoder as soon as vLLM has written it, instead of after the whole prefill. For a 70B model at 32k tokens the cache is about 10.7 GB in fp16 (134 MB per layer, about 0.11 s per layer at 10 GbE line rate), while a DGX Spark spends about 0.5 s computing each layer, so most of the transfer can hide behind the prefill. The DeepSeek V4 path exports from `get_finished()`, after the request has run, so it has nothing to overlap and refuses the flag.

Measured with Llama-3.3-70B (NVFP4 prefill on one DGX Spark, 4-bit MLX decode on one Mac Studio, 10 GbE, prompts of about 27k tokens, warm store): first token 60.0 to 63.1 s buffered, 50.0 s with prefill-side streaming, 48.7 s with streaming at both ends. The streamed transfer finished within 0.1 s of vLLM's prefill, and streaming did not slow the prefill.

## 1. Design in one page

```
 vLLM forward pass (TP rank 0)                transfer service / bridge               decoder (EXO)
 ------------------------------                -------------------------               -------------
 begin(tag, stream=descriptor)   ---- streaming/<tag>, stream.log line 1 ---->
 layer 0: gather, all-gather, copy_ into L0.bin
          stream.log += {"layer": L0}          poll /stream: layer 0 ready
                                               GET layers/0 (X-KVH-Live)      Header, KVChunk 0 --> decode, stage
 layer 1: ...  stream.log += {"layer": L1}     GET layers/1                   KVChunk 1  -------> decode, stage
 ...                                           ...
 layer 79      stream.log += {"layer": L79}    GET layers/79                  KVChunk 79 -------> decode, stage
 wait_for_save: header.json, tags/<tag>,
          stream.log += {"state": "published"} published, POST returned 200,
                                               layers match                   Done  ------------> commit to cache
```

The decisions:

- **The wire format does not change.** Header, one KVChunk per layer in index order, then Done, exactly as today. Only the timing changes.
- **Done is the commit, on both sides.**
  - The bridge sends Done only when three things hold: the export is published, vLLM's request returned 200, and the published layer list equals what was sent.
  - The Mac writes nothing into its cache until Done.
  - Any failure after the first byte ends the stream with an EXO error frame.
- **The published form of an export does not change.** `header.json` and `tags/<tag>` are still written only at publish. Progress goes through a separate surface: a `streaming/<tag>` link and the slot's `stream.log`. A reader that knows nothing about streaming (an old bridge, the dsv4 bridge) can never see a partial export.
- **Either flag works without the other.**
  - A streaming bridge facing a non-streaming connector sees the published export first, and sends it the buffered way.
  - A streaming connector facing an old bridge looks like today's export to that bridge.
  - EXO's per-layer ingest works with any sender.
  - So the three can be enabled one at a time.

## 2. Prefill side (vLLM, transfer service, bridge)

### 2.1 How the bridge learns that layer N is complete

There is a per-slot progress log, `slots/s<k>/stream.log`, written as JSON lines. It is used only in streaming mode.

| Line | Written by | When |
| --- | --- | --- |
| `{"stream": 1, "header": {...}}`, the descriptor: kind, tag, model, token count, `num_model_layers`, RoPE, producer, and the seq the export will be published with | `begin(tag, stream=...)` | before any layer, while `begin()` still holds the slot's `LOCK_EX` |
| `{"layer": {...}}`, the layer's `LayerRef` (file, offsets, shapes, dtype) | `stream_layer()` from `PagedWriter.write_layer` | after the layer's bytes are in the file (section 2.2) |
| `{"state": "published", "seq": n}` | `publish()` | after `header.json` and `tags/<tag>` |
| `{"state": "failed", "error": ...}` | `stream_failed()`, or `abandon()` | as soon as the writer gives the export up |

Readers use two calls:

- `ExportReader.stream_state(tag)` parses the log and returns the descriptor with the layers completed so far, in index order. It returns the final header once the export is published, and raises `ExportFailed` for a failed stream or a failure record.
- `open_stream(tag, seq)` holds the slot with `LOCK_SH` and allows reads of completed layers only.

Over HTTP, the transfer service exposes these as `GET /v1/exports/<tag>/stream` and `GET /v1/exports/<tag>/layers/<i>` with `X-KVH-Live: 1`.

The log is a set of completed layers, not a counter. So layers written out of order (any future layout that is not 0..N-1 in order) still work: the bridge sends them in index order and waits for the next index it needs.

Each line costs one `open`, `flock`, `write`, `unlock` and `close` in the vLLM worker, tens of microseconds. That is about 80 lines per export, against about 40 s of prefill.

### 2.2 Memory ordering

The writer and the reader are different processes: vLLM, and the transfer service or the bridge. GB10's CPU is Arm (Grace), so the argument cannot rest on x86's strong ordering. It has three links.

1. **The bytes are complete before the writer moves on.**
   - `write_layer` copies the layer into the slot's shared file map with `dst.copy_(kv_split)`.
   - The destination is pageable host memory, and `non_blocking` is False. So the copy is host-synchronous: `copy_` returns only once the device-to-host copy has finished and the data is visible to the calling thread.
   - Whether CUDA stages the copy through a pinned buffer or GB10's coherent C2C link writes it directly, the CUDA runtime's completion wait is an acquire on the calling thread.
   - In streaming mode the writer also calls `torch.cuda.current_stream().synchronize()` (`_device_fence`) before announcing the layer. Today that is free, because `copy_` has already waited. It keeps the guarantee if the copy is ever made non-blocking.
2. **The announcement is a release on a lock the reader acquires.**
   - The log line is appended under `flock(LOCK_EX)` on `stream.log` and followed by `flock(LOCK_UN)`.
   - The reader reads the log under `flock(LOCK_SH)`, which it can only take after the writer's unlock.
   - In Linux, `flock` takes and drops the same per-inode spinlock (`flc_lock`) for both lock and unlock. The writer's unlock is a release, the reader's lock is an acquire on the same lock, and that gives happens-before in the Linux kernel memory model.
   - So everything the writer did before the unlock is visible to the reader after its lock. That includes the layer's stores into the page-cache pages of `L<i>.bin`, and the `posix_fallocate` that allocated them.
3. **The reader reads the bytes only after it acquired the log.**
   - `open_stream` takes the slot's `LOCK_SH`, then reads `stream.log` under its own `LOCK_SH` and checks the tag and seq. It then serves only the layers listed there, through `pread` or `sendfile` of the same tmpfs pages, which are coherent between the writer's `MAP_SHARED` mapping and the reader's file reads.
   - Over HTTP, the server thread that sends a layer runs `open_stream` itself, so it acquires the log on its own and does not rely on the earlier `/stream` poll.

A layer is never rewritten within one stream: `PagedWriter` fails the export if the same layer index is written twice. Once a reader has seen a layer, its bytes cannot change until the slot is reused, and reuse is covered in section 2.4.

### 2.3 Streaming while vLLM is still running

The bridge (`_serve_paged_stream`) works in this order:

1. **POST in the background.** It sends vLLM's request on a background thread (`_posts`, sized by `max_inflight`). The in-flight semaphore is released when the request completes, so `max_inflight` still bounds vLLM prefills.
2. **Wait for the first layer.** It polls `stream_state` (2 ms, backing off to 20 ms) until any layer is listed.
   - If the first state it sees is already published (a non-streaming connector), it uses the published layer list.
   - Otherwise it expects layers `0..num_model_layers-1`.
3. **Fetch layer 0 and check it.** It fetches layer 0 and checks the kind, model, token count and wire dtype. Only then does it write EXO's Header. Every failure before this point still raises before the first byte, as today.
4. **Stream the layers.** For each layer in index order, it sends the KVChunk while the next layer is awaited and fetched on the fetch pool (one ahead, as in the buffered path).
   - Each wait checks three things at every poll: vLLM's request (a failed POST stops the stream at once), the stream's failure line, and a stop flag set when the hand-off itself fails.
5. **Commit.** It waits for the published state and for the POST to return. It checks that the published seq and layer list equal what it streamed, and only then sends Done and releases the export.

Bytes on the wire to EXO are identical to the buffered bridge's for the same export. The tests compare them byte for byte.

### 2.4 Abort, slot reuse, timeouts

**What "abort mid-prefill" can mean on the vLLM side.**

| Case | What the store sees | What the bridge does |
| --- | --- | --- |
| Request aborted (client gone, cancel) before its final prefill step | Nothing: `RequestTracker` emits a job only in the step that completes the prompt, and drops aborted requests | POST fails, so it raises before the first byte |
| Abort during the final step | vLLM processes aborts between steps, so the step's forward runs to completion and the export is published | Streams normally |
| Writer error at layer k (ENOSPC, wrong row count, unsupported layout, a layer written twice) | `stream.log` gets `failed` at once and a failure record is written at once (not at `wait_for_save`). The slot is abandoned at `wait_for_save` | Raises at its next poll (2 to 20 ms) and sends an error frame |
| vLLM worker or engine dies mid-forward | The log stops with no terminal line. The writer's flocks die with the process, and the restarted store drops every `streaming/` link (`_recover`) | The POST fails (connection reset or 500), so it raises at its next poll. Failing that, the per-layer timeout applies. It sends an error frame |

**Slot reuse.**

- A streaming slot stays in the writer's `_writing` set from `begin()` to `publish()` or `abandon()`, so `begin()` can never pick it.
- The writer does not hold the slot's `LOCK_EX` during the stream (it is released at the end of `begin()`). That is what lets readers in.
- A reader's `LOCK_SH` still blocks reuse later: `begin()` needs `LOCK_EX` without blocking, and skips a slot a reader holds. This is tested with a failed stream whose only slot is held by a reader.
- **The seq is fixed at `begin()`** and kept by `publish()`. Every layer read is checked against it, by `open_stream` and by `X-KVH-Seq` over HTTP. So a slot reused between two reads (after abandon, or after `hold_seconds`) is detected and never serves another export's bytes.
- `begin()` on a reused slot removes the old `streaming/` and `tags/` links and the old `stream.log` while it holds `LOCK_EX`.

**Timeouts.**

- `timeout_s` (default 50 s) still covers the pre-header phase: vLLM's queue and prefill up to layer 0, and fetching layer 0.
- vLLM's POST keeps the same whole-prefill deadline it has today, so streaming adds no new limit.
- After the Header, each further layer gets its own `timeout_s`, as the buffered path does for fetches.
- The gap between KVChunks is about one layer of compute (0.5 s at 32k), far below EXO's 60 s per-receive socket timeout.
- The final wait for publish gets its own `timeout_s`.

### 2.5 `wait_for_save` and tensor-parallel ranks

**`wait_for_save`.** It is unchanged: it synchronises the device and calls `finish_step()`, which publishes the export. In streaming mode, publishing appends the `published` line. Streaming does not depend on `wait_for_save` for any layer, only for Done.

**Tensor parallel.**

- Each TP rank gathers and normalises its own head slice in `save_kv_layer`.
- All ranks all-gather the head axis for every job in the same order, and only rank 0 writes.
- The all-gather is a collective, so rank 0 cannot hold layer N's gathered tensor until every rank has contributed its slice of layer N.
- So **a layer announced by rank 0 is complete across all ranks by construction**, and no per-rank progress is needed. Ranks other than 0 never touch the store.
- `test_tp_layer_is_announced_only_once_every_rank_has_contributed` runs two ranks as two threads with a blocking all-gather. It holds rank 1 back at each layer, checks that layer N is not in the log until rank 1 reaches it, and checks that the announced bytes hold all heads.

**Alternative, not built.** Each rank could write its own head slice straight into the file, which saves the all-gather traffic. Then a layer would be complete only when every rank has logged it: the log would carry `{"layer": i, "rank": r}` lines, and readers would wait for `world_size` of them. The all-gather path is the hardware-tested one, so it stays.

### 2.6 When the overlap happens: chunked prefill

A `StoreJob` is emitted only in the step that completes the prompt. The overlap window is therefore the forward pass of the **final** prefill step.

- The hardware runs used `--max-num-batched-tokens = max_model_len + 1024`, so a 32k prompt is one step and the whole prefill overlaps the transfer.
- With chunked prefill (for example `--max-num-batched-tokens 8192`), only the last chunk overlaps.
- The gain holds as long as the final step's compute per layer is at least the transfer time per layer.
  - At 32k with 8k chunks, the last chunk attends over the whole context and is the most expensive. That is about 0.16 s per layer if it is a third of the total, still more than 0.11 s.
  - With small chunks at long context, the transfer becomes network-bound inside the last step .
- **Possible extension, not built.** EXO's client already concatenates several KVChunks per layer in arrival order, so the bridge could send each step's rows as soon as that step finishes, again without a wire change. That needs per-step row ranges from the connector, and it changes EXO's progress callback (which assumes one chunk per layer).

## 3. Decode side (EXO)

This half lives in EXO, not in this package: a per-layer ingest path in EXO's disaggregated-prefill client, behind `EXO_LAYERWISE_REMOTE_INGEST` (off by default), proposed to EXO separately. The wire format is unchanged, so kvhandoff works with either EXO client.

### 3.1 Per-layer ingest

`LayerwiseIngest` (`exo/worker/engines/mlx/disaggregated/client.py`) works in four steps:

1. **Decode on arrival.** When a KVChunk arrives, it is decoded at once into MLX arrays (`chunk_to_mlx_nhd`) and evaluated, so its wire bytes are dropped. An `ArraysState` is decoded the same way.
2. **Stage, do not write.** Staged layers are kept in a dict. **Nothing is written into the caller's caches.**
3. **Commit at Done.** `commit(caches, start_pos)` requires Done. It applies every staged layer with the same calls `ingest_into_mlx_cache()` makes:
   - several chunks of one layer are concatenated on the token axis;
   - `RotatingKVCache` gets `final_offset`;
   - `KVCache` with `start_pos > 0` keeps its prefix.
   So the cache is identical to buffered ingest (tested for `KVCache`, `RotatingKVCache` and `ArraysCache`, with `start_pos` 0 and 3, one layer split over two chunks, layers out of order).
4. **All or nothing.** If applying any layer raises (for example a head-count mismatch with the prefix), the layers already applied are restored from snapshots and the error propagates.

`remote_prefill_fetch(..., layerwise=)` hands it each message. With `layerwise`, a stream that ends without Done raises. The stock client takes EOF at a frame boundary as a complete stream.

**What it gains.**

- The decode work overlaps the transfer instead of following it. The prototype logged `inject=` 0.3 to 1.0 s for 70B at about 27k.
- Peak memory falls by roughly the size of the stream: about one layer of wire bytes in flight instead of 10.7 GB of bytes beside 10.7 GB of arrays at 32k.

### 3.2 Mid-stream failure

The flow is:

1. The fetch raises: on an error frame (`RuntimeError`), a closed socket before Done (`ConnectionError`), or a truncated frame.
2. The staged arrays are dropped with the `LayerwiseIngest` object.
3. The caller (`BatchGenerator.submit`, `mlx_generate`) catches the exception and prefills locally from the same `cache` object, which was never written to.

Tests cover all three failures, after two of four layers, with a prefix cache present. A layer that has been only partly ingested cannot exist in the cache, because no layer is ingested before Done.

## 4. Wire format

**It does not change.** Header, then KVChunk per layer, then Done. These already arrive in order on one TCP stream, and EXO's client already accepts any number of KVChunks per layer in arrival order.

Two behaviours are tightened without changing it:

- **The bridge's error frame.** It now writes EXO's error frame before raising after the first byte, in streaming and buffered paged modes alike.
  - EXO's `PrefillServer` already follows a raising resolver with a Header and an error frame. A client mid-stream fails to decode that Header as a message and raises, so the fallback worked before too, but by accident.
  - The explicit error frame makes it deliberate, and it closes the case where a stream cut at a frame boundary passes for a complete one.
- **Per-layer ingest requires Done.** The stock buffered client does not, and this is listed in the PR as a follow-up.

