# DeepSeek V4 receiver cross-check (two interpreters)

kvhandoff runs with torch; the receiver of EXO's DeepSeek V4 cache transport
([exo-explore/exo#2545](https://github.com/exo-explore/exo/pull/2545)) needs MLX. This check spans both:

```bash
# 1. kvhandoff's DeepSeek V4 blobs for T in (5, 300, 1030) x compress ratio (0, 4, 128), from the fake vLLM layouts
<torch venv>/bin/python tests/compat/dump_blobs.py "$PWD" /tmp/kvh-blobs
# 2. The PR's inject_deepseek_v4_cache rebuilds a DeepseekV4Cache from them; its deepseek_v4_cache_to_blobs must give
#    back the same bytes, and the cache offset must equal T
PYTHONPATH=<EXO checkout with PR 2545 applied>/src MLX_DISABLE_COMPILE=1 \
  <venv with mlx + EXO's mlx-lm pin>/bin/python tests/compat/ingest_exo_dsv4.py /tmp/kvh-blobs
```

Result (MLX CPU 0.31.2, mlx-lm 6a3df6cd): `DSV4-COMPAT 9/9`.
