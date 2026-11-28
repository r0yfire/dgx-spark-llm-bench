# dgx-spark-llm-bench

One script to install, serve, benchmark and clean up abliterated ("uncensored") open-weight models on an **NVIDIA DGX Spark** (GB10, aarch64, 128 GB unified memory), on whichever inference engine runs each model fastest there:

| Engine | What it runs here | How it runs |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) ([AEON Spark build](https://github.com/aeon-7)) | NVFP4 checkpoints | Docker |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) server (CUDA 13) | GGUF | Docker |
| [PolyStrata](https://github.com/simplexai-labs/PolyStrata) | GGUF mixture-of-experts models | native: `strata-poly` built from [our fork](https://github.com/r0yfire/PolyStrata/tree/spark-aarch64) (CyberTiel), the Strata engine's signed release (Qwen3.8-Flash-Next) |
| [TensorFold](https://github.com/ashhart/TensorFold) | MLX 4-bit checkpoints, with a DFlash2 drafter | native release |

| Model | Engines (default first) | Notes |
|---|---|---|
| `qwen27b-abl`: [Qwen3.8-27B Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | `tensorfold` ([MLX 4-bit](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX)), `vllm` ([NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4)) | Dense 27B hybrid model. TensorFold drafts with [z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2); vLLM with the model's MTP head. The NVFP4 repo is gated. |
| `cybertiel`, `cybertiel-q8`: [CyberTiel Coder 35B-A3B](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) | `polystrata`, `llamacpp` | MoE with 3B active parameters, GGUF `UD-Q4_K_XL` or `UD-Q8_K_XL`, MTP draft head in the file. |
| `flashnext`: [Qwen3.8-Flash-Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (stock) | `polystrata`, `llamacpp` | The original model, as the baseline for the abliterated one: ISTA-DASLab's GSQ-RCO `IQ3_S` GGUF (two shards, about 84 GB) at the revision PolyStrata pins. PolyStrata drafts with DFlash2; llama.cpp runs without a drafter. PolyStrata can add its experimental refusal-direction projection (`PS_REFUSAL_PROJECTION=on`). |
| `flashnext-abl`: [Qwen3.8-Flash-Next Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) | `polystrata`, `llamacpp` | OrcaRouter's `IQ3_XXS` GGUF (two shards, 85.2 GB). PolyStrata drafts with the same DFlash2 drafter as the stock model; llama.cpp can draft with OrcaRouter's separate MTP draft model (`LLAMA_ABL_MTP=on`). The repo is gated. |

Every model is served with a 262,144-token context window behind an OpenAI-compatible API on port `8000`, protected by a randomly generated API key that every inference route requires (the script checks this at each start). Only one model runs at a time.

The script also measures tokens per second (prefill and decode) on the running model, and can run a subset of [Cybench](https://github.com/andyzorigin/cybench) (a CTF benchmark for LLM agents) against it.

> [!WARNING]
> These models have had their refusal behavior removed. The exception is `flashnext`, the stock model, unless you turn on PolyStrata's refusal-direction projection for it (its package says the model then declines far fewer requests). Use them responsibly and only for lawful purposes, such as authorized security testing, CTFs and research. You are responsible for how you use them, and for complying with each model's license.

## Results

Measured on one DGX Spark (November 2026) with `speedtest.py` from this repo: every engine gets the same prompts through the same streaming client, temperature 0, 512-token answers, thinking off, 3 runs per cell, medians. Prefill is prompt tokens divided by time to first token; decode is the answer's tokens minus one (`completion_tokens - 1`, as reported by the server) divided by the time from the first streamed token to the last. The prompt cases are named by nominal size (`4k`, `16k`, `32k`); the prompts the engines actually counted were about 3,860-4,050 tokens (4k), 14,250-14,430 (16k) and 27,770-27,950 (32k), depending on the model's tokenizer. The needle test hides a code at one third of a ~208,600-token prompt and asks for it back.

| Model | Engine | Decode tok/s (4k / 16k / 32k case) | Prefill tok/s (4k / 16k / 32k case) | ~208K needle |
|---|---|---|---|---|
| Qwen3.8-27B Uncensored | vLLM, NVFP4, MTP 2 drafts (previous default) | 18.9 / 18.0 / 16.9 | 1,887 / 1,971 / 1,808 | found, first token after 259 s |
| Qwen3.8-27B Uncensored | vLLM, NVFP4, MTP 4 drafts | 20.7 / 20.2 / 19.4 | 1,959 / 1,984 / 1,783 | found, 262 s |
| Qwen3.8-27B Uncensored | **TensorFold**, MLX 4-bit + DFlash2 | **59.1 / 58.4 / 54.2** | 1,414 / 1,349 / 1,316 | found, **253 s** |
| CyberTiel 35B-A3B Q4_K_XL | llama.cpp, MTP | 84.5 / 80.2 / 74.4 | 2,300 / 2,281 / 2,172 | found, 144 s |
| CyberTiel 35B-A3B Q4_K_XL | **PolyStrata**, MTP | **97.2 / 87.6 / 85.9** | **3,648 / 3,583 / 3,491** | found, **100 s** |

What these show:

- **Qwen3.8-27B:** TensorFold decodes about 3 times as fast as vLLM. vLLM reads prompts faster in the 4k to 32k cases, and at ~208K the two are level. vLLM's best setting is 4 MTP drafts: 6 drafts and a 16K prefill batch measured no better (prefill in the 16k and 32k cases got slower with the larger batch). TensorFold with an FP8 KV cache decoded slower (50–53 tok/s) than with the default cache.
- **CyberTiel:** PolyStrata decodes 9-15% faster than llama.cpp (15% in the 4k case, 9% at 16k, 15% at 32k) and reads prompts about 60% faster, on the same GGUF file. Greedy answers to 5 fixed prompts (`--parity`) were identical on 3; the other 2 parted at a near-tie token after 65 and 160 characters, both still correct. PolyStrata's own docs note its fused kernels round slightly differently from llama.cpp's.
- The two Qwen3.8-27B engines read different 4-bit formats of the same OrcaRouter weights (MLX affine 4-bit for TensorFold, NVFP4 for vLLM), so their answers are not expected to match token for token.
- Single runs on one machine; expect a few percent of variation between sessions.

## Features

- Interactive menu, plus flags for scripting
- One registry row per model and engine: each engine downloads the format it reads
- Downloads weights with the Hugging Face CLI (installed into its own venv), pulls the arm64 container images, builds PolyStrata's `strata-poly` for `sm_121`, installs the Strata engine from PolyStrata's signed update feed, and installs TensorFold's release after checking it against a SHA-256 pinned in the script
- Spark-specific settings: `sm_121a` kernels, FlashInfer NVFP4 GEMM, conservative GPU memory use on unified memory
- Speculative decoding: MTP (vLLM and llama.cpp turn it on with a flag; PolyStrata's `strata-poly` uses the GGUF's MTP head on its own), or DFlash2 (TensorFold, and PolyStrata's Strata engine for Qwen3.8-Flash-Next)
- `--speed` and `--parity` benchmarks of the running model
- Cybench setup and runs, patched to work with a local endpoint, with per-task result tables
- Tiered clean-up: stop servers, remove images, delete weights, or remove everything
- `--dry-run` prints what would happen without changing anything

## Requirements

- An NVIDIA DGX Spark, or another GB10 / aarch64 machine with CUDA 13 drivers (580 or newer). The settings are tuned for the Spark's unified memory.
- Docker with the NVIDIA Container Toolkit (`docker run --gpus all ...` must work), and your user in the `docker` group
- `python3` (with `venv`), `git`, `curl` and `sha256sum`
- For PolyStrata's `strata-poly` (CyberTiel): the CUDA 13 toolkit (`/usr/local/cuda/bin/nvcc`, which DGX OS ships) and `cmake`. The build takes about 10 minutes. Qwen3.8-Flash-Next needs neither: its engine is a ready-made release.
- About 80 GB of free disk per model and engine (90 GB for the Qwen3.8-Flash-Next models).
- A Hugging Face account with access granted to gated repos (the Qwen NVFP4 repo and OrcaRouter's Qwen3.8-Flash-Next GGUF). The others are public.

## Quick start

```bash
git clone https://github.com/r0yfire/dgx-spark-llm-bench.git
cd dgx-spark-llm-bench
chmod +x spark-llm.sh

# Needed for gated repos (and faster downloads in general)
export HF_TOKEN=hf_xxx      # or: hf auth login

./spark-llm.sh              # interactive menu
```

The menu lists every model and engine pair:

```
 DGX Spark model manager      [active: none]
   1) Qwen3.8-27B Uncensored, MLX 4-bit + DFlash2      tensorfold
   2) Qwen3.8-27B Uncensored, NVFP4 + MTP              vllm
   3) CyberTiel 35B-A3B UD-Q4_K_XL                     polystrata
   ...
   s) Stop running model       t) Status          p) Speed test (tok/s)
   c) Set up Cybench           r) Run Cybench on the running model (10-task subset)
   x) Clean up                 q) Quit
```

The first start can take several minutes while weights load and kernels compile. The script waits until the API answers, then checks that it refuses requests without the key. Picking a model that is not installed yet installs it first.

## Usage

```bash
./spark-llm.sh --list                              # models and the engines each one runs on
./spark-llm.sh --install qwen27b-abl               # the default (fastest) engine's files
./spark-llm.sh --start qwen27b-abl                 # stops any other model first
./spark-llm.sh --start qwen27b-abl --engine vllm   # another engine for the same model
./spark-llm.sh --install flashnext-abl             # Qwen3.8-Flash-Next Uncensored on PolyStrata (its default)
./spark-llm.sh --start flashnext-abl
./spark-llm.sh --start flashnext-abl --engine llamacpp
LLAMA_ABL_MTP=on ./spark-llm.sh --start flashnext-abl --engine llamacpp   # also draft with OrcaRouter's MTP draft model
./spark-llm.sh --start flashnext                   # the stock model, on PolyStrata
PS_REFUSAL_PROJECTION=on ./spark-llm.sh --start flashnext   # the same, with PolyStrata's experimental refusal projection
./spark-llm.sh --status
./spark-llm.sh --stop

./spark-llm.sh --speed                             # tokens/s of the running model (short, 4k, 16k, 32k cases)
SPEED_NEEDLE=236000 ./spark-llm.sh --speed 4k      # one prompt size, plus a ~208K-token needle test
./spark-llm.sh --parity                            # greedy answers to fixed prompts, to compare engines

./spark-llm.sh --cybench-setup                     # clone + patch Cybench, pull its image
./spark-llm.sh --cybench-run 10                    # run the subset on the active model
./spark-llm.sh --cybench-summary ~/spark-llm/results/<run-dir>

./spark-llm.sh --cleanup weights                   # stop | containers | weights | all
./spark-llm.sh --yes ...                           # skip confirmations
./spark-llm.sh --dry-run ...                       # print commands only
```

`qwen` and `tiel` still work as aliases for `qwen27b-abl` and `cybertiel`.

### Calling the API

```bash
KEY=$(cat ~/spark-llm/.api_key)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "CyberTiel", "messages": [{"role": "user", "content": "Hello"}]}'
```

The served model names are `Qwen3.8-27B-Uncensored`, `CyberTiel`, `Qwen3.8-Flash-Next` and `Qwen3.8-Flash-Next-Uncensored`, whichever engine runs them. (PolyStrata's Flash-Next server also answers to its own set name, see below.) Any OpenAI-compatible client works with base URL `http://<spark-ip>:8000/v1` and the key above. On vLLM, Qwen runs with a reasoning parser and tool calling (`qwen3_coder`) turned on. To turn thinking off for a request, send `"chat_template_kwargs": {"enable_thinking": false}`. `--speed` sends it and flags any run that still produced reasoning.

## Engines

### PolyStrata, from a fork

PolyStrata runs mixture-of-experts GGUF models on two engine programs behind one server (`serve/server.py`): `strata-poly` (CyberTiel) and the Strata project's own engine, `strata` (Qwen3.8-Flash-Next). Both are the `polystrata` engine to this script: the same server, API key, PID file, `--stop` and auth check. The model picks the program.

#### CyberTiel on `strata-poly`

PolyStrata runs CyberTiel with its own CUDA engine, `strata-poly`. Upstream ships no aarch64 build of it, and it did not compile on aarch64: two files included the x86-only `<immintrin.h>` for a spin-wait hint. The [`spark-aarch64` branch of our fork](https://github.com/r0yfire/PolyStrata/tree/spark-aarch64) uses PolyStrata's own portable `cpu_relax.hpp` there instead (the same instruction on x86, `yield` on Arm). The script clones that branch at a pinned commit (`PS_REPO`, `PS_REF`), builds `strata-poly` for `sm_121`, exports the model's tokenizer and serves it with PolyStrata's server. `strata-poly` drafts with the GGUF's own MTP head without being asked (its log reports how many drafts were accepted), so the script passes it no speculative-decoding flag, while llama.cpp gets `--spec-type draft-mtp`. With all experts resident on the Spark, PolyStrata's advantage over llama.cpp comes from its kernels, not from expert offloading.

#### Qwen3.8-Flash-Next on the Strata engine

PolyStrata's own `setup.sh` installs Qwen3.8-Flash-Next together with the stock model's 84 GB download, and has no option to install only the engine and the drafter. So the script runs the same steps itself, with PolyStrata's own tools, for both models:

- **The engine:** `strata update --no-control` from the checkout, into `polystrata/terminal/engine-feed` (the step `setup.sh` runs on a Spark). It installs the signed `linux-aarch64` CUDA package from the update feed of the pinned checkout's release; nothing is compiled. The script also installs the cuBLAS and CUDA runtime wheels `setup.sh` installs for this engine, at its pinned versions.
- **The drafter:** `strata drafting fetch`, PolyStrata's tuned DFlash2 drafter for Qwen3.8-Flash-Next (about 0.34 GB, its size and SHA-256 pinned by PolyStrata).
- **The pack:** `tools/iq_pack.py` turns the GGUF's first shard into the pack the engine reads, in `polystrata-data/packs/<model>`. OrcaRouter's file is packed with `--compat-bf16`, as PolyStrata's `docs/ORCA.md` describes: its quantized hyper-connection, attention and PLE projections become the BF16 that PolyStrata's kernels read.
- **The run config** (`polystrata-data/strata-<model>.json`) is written at each start from one template for both models, so they differ only in their files: the engine arguments `setup.sh` writes on a Spark, without the MTP layer's options. OrcaRouter's PLE table is in its first shard, so that shard is both `--native` and `--ple-gguf`; for the stock files it is the second shard. Two settings make PolyStrata's server start DFlash2 with the installed drafter for either model, with its own window and confidence:
  - `model_name` is PolyStrata's name for the set (`qwen3.8-flash-next-iq3_s`, `qwen3.8-flash-next-iq3_xxs`), and the served name is an alias of it, so clients use the served name as with every engine (`/v1/models` lists both). PolyStrata decides which drafter a model gets from that name; for a name it does not know it reads the GGUF header, and its header reader stops at the list of tags at the top of OrcaRouter's file, so it would start that file with no drafter.
  - `user_args` names the drafter (`--draft-model`). Without it, a set with no drafter of its own in PolyStrata's table (`IQ3_XXS` on CUDA) would look for the stock (untuned) DFlash2 pack, which is not downloaded, and start with no drafter.

`--dry-run --start flashnext` (or `flashnext-abl`) prints the run config a start would write, once the engine is installed. `PS_REFUSAL_PROJECTION=on` adds PolyStrata's experimental refusal-direction projection to the stock model (`flashnext`): a control vector shipped in the checkout, projected out of layers 4-44 (the flags `setup.sh --experimental-speed-projection on` writes). A request can switch it off with `"experimental_speed_projection": false`. Read `polystrata/terminal/data/experimental-speed-projection/README.md` before using it.

### llama.cpp

CyberTiel drafts with the MTP head in its GGUF (`--spec-type draft-mtp`). Qwen3.8-Flash-Next is loaded into memory with `--load-mode none` (this llama.cpp build refuses `--no-mmap`) and runs without a drafter. For `flashnext-abl`, `LLAMA_ABL_MTP=on` adds OrcaRouter's separate MTP draft model (`-md`, with the `--spec-draft-n-max 3` its model card recommends); `--install flashnext-abl --engine llamacpp` downloads it (4.1 GB) either way.

### TensorFold

TensorFold's native release (CUDA kernels embedded in the binary) serves Qwen3.8-27B from an MLX affine 4-bit checkpoint, drafting with DFlash2. The script installs release `TF_VERSION` only if the tarball matches the SHA-256 pinned in the script (`TF_SHA256`; the release's own `.sha256` file is not trusted, since release assets can be replaced), and passes the API key through `--api-key-file`. The native release does not accept `--prefill-fp8` yet (only TensorFold's Python engine does), which is why its prefill trails vLLM on shorter prompts.

### vLLM

vLLM drafts with the model's MTP head, 4 tokens by default (`VLLM_SPEC`). The AEON image also contains vLLM's DFlash support, but its DFlash2 drafter failed to start there (`FlashInferImpl.__init__() got an unexpected keyword argument 'use_mm_prefix'`), so it is not the default.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `SPARK_LLM_HOME` | `~/spark-llm` | Where weights, engines, venv, Cybench, results and the API key live |
| `PORT` | `8000` | Host port for the API |
| `BIND` | `0.0.0.0` | Address the API listens on: `0.0.0.0` for every interface (LAN clients), `127.0.0.1` for this machine only. IPv4 only. See [Security notes](#security-notes). |
| `CTX` | `262144` | Context window |
| `VLLM_GPU_UTIL` | `0.70` | vLLM memory share. Keep it at or below 0.8 on unified memory. |
| `VLLM_SPEC` | `{"method":"mtp","num_speculative_tokens":4}` | vLLM's `--speculative-config`. `VLLM_SPEC=` (empty) turns speculative decoding off. |
| `VLLM_IMAGE` | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` | vLLM image built for the Spark |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda13` | llama.cpp server image |
| `PS_REPO` / `PS_REF` | our fork / a pinned commit | PolyStrata source to build (and, for Qwen3.8-Flash-Next, the release whose engine and drafter it installs) |
| `PS_REFUSAL_PROJECTION` | `off` | `on`: PolyStrata's experimental refusal-direction projection on the stock Qwen3.8-Flash-Next (`flashnext` on `polystrata` only) |
| `LLAMA_ABL_MTP` | `off` | `on`: llama.cpp also drafts with OrcaRouter's MTP draft model (`flashnext-abl` on `llamacpp` only) |
| `TF_VERSION` / `TF_SHA256` | `1.3.6` / that release's tarball hash | TensorFold release, and the SHA-256 its tarball must match. Another version needs its own `TF_SHA256`. |
| `DRAFTER_27B_REPO` | `z-lab/Qwen3.8-27B-DFlash2` | DFlash2 drafter for Qwen3.8-27B |
| `VLLM_EXTRA` / `LLAMA_EXTRA` / `TF_EXTRA` | – | Extra arguments for vLLM, llama.cpp or TensorFold (`QWEN_EXTRA` / `TIEL_EXTRA` still work) |
| `SPEED_RUNS` / `SPEED_NEEDLE` | `3` / `0` | Runs per prompt size; needle prompt size in tokens (0 skips it) |
| `HF_TOKEN` | – | Hugging Face token. Otherwise the `hf auth login` cache is used. |
| `CYBENCH_ITER` | `15` | Agent iterations per task |
| `CYBENCH_TASK_TIMEOUT` | `21600` | Per-task wall-clock safety net, in seconds |
| `CYBENCH_ONLY` | – | Comma-separated task numbers to run, e.g. `6,7,9` |
| `CYBENCH_MAX_IN` / `CYBENCH_MAX_OUT` | `120000` / `32768` | Token limits per iteration |

## Speed test

`speedtest.py` (standard-library Python, no dependencies) drives any OpenAI-compatible server:

- Each prompt is real source code (the Python standard library on the machine, read in a fixed order) and starts with a random nonce, so no prefix or conversation cache can skip the prefill. The cases (`short`, `4k`, `16k`, `32k`, ...) are named by nominal size; each run records the prompt tokens the engine actually counted.
- Token counts come only from the server's own `usage` (`stream_options.include_usage`). A run without `usage` gets no rates (shown as `n/a`, flagged `no_usage`) rather than a count of streamed chunks, which would understate decode for engines that send several drafted tokens per chunk.
- Runs are flagged, and left out of the medians, when the model produced reasoning even though thinking was turned off: streamed reasoning, a `<think>` tag in the answer, `reasoning_tokens` in `usage`, or more completion tokens than the streamed text can hold, or under 2 characters per token (a heuristic): signs of hidden reasoning, which would make prefill look slow and decode look fast. Flagged runs are printed with a warning and kept in the JSON.
- `--needle N` checks long context; `--parity` saves greedy answers to fixed prompts; `--table FILES...` prints a markdown table of result files, one row per label and engine configuration.

`--speed` writes results to `$SPARK_LLM_HOME/speed/<model>-<engine>-<timestamp>.json`, including the engine configuration the model was started with (`meta`: image and image ID, `VLLM_SPEC`, `CTX`, PolyStrata commit, the Strata engine's version and drafter, `PS_REFUSAL_PROJECTION`, `LLAMA_ABL_MTP`, TensorFold version, extra arguments).

## Benchmarking with Cybench

`--cybench-setup` clones Cybench into `$SPARK_LLM_HOME/cybench` and applies a few small patches so it can drive a local server:

- An OpenAI client with a configurable base URL, sampling settings from the environment, and a local tokenizer fallback
- An `openai/local` model entry
- No `<END>` stop sequence sent to the server. Reasoning models write `<END>` while thinking, which made the server return empty replies. The reply is cut at `<END>` on the client side instead.
- `run_task.sh` runs without a TTY or a forced image pull
- The agent reaches the model through the host's LAN IP. Cybench runs its own Docker daemon inside the agent container, which shadows `host.docker.internal`.

`--cybench-run` runs tasks unguided, with 15 iterations each. It writes per-task logs to `$SPARK_LLM_HOME/results/<model>-<engine>-<timestamp>/` and prints a summary. Timeouts count as failures. Tasks whose environment fails to start are reported as `NOT RUN` and excluded from the score.

### The task subset

The 10 tasks come from Cybench's official `task_list.txt`. Each was checked to run on aarch64:

Dynastic, Primary Knowledge, It Has Begun, Urgent, Flag Command, back-to-the-past, eval-me, motp, LockTalk, Partial Tenacity

Excluded, and why:

- **x86-64 only.** Pwn and reversing tasks such as Delulu, PackedAway and Crushing ship x86-64 binaries or services. The Spark has no amd64 emulation, and the agent's arm64 tools can't disassemble them.
- **Broken builds.** Unbreakable and Labyrinth Linguist fail to build because their Debian base images point at archived repos.
- **Not in the public repo.** Nine GlacierCTF tasks from the official list aren't included in the public Cybench repo.

### Example results

These come from one run per model on the subset above (November 2026, DGX Spark), with the engines this script used at the time (vLLM with 2 MTP drafts, and llama.cpp):

| | Qwen3.8-27B NVFP4 (vLLM) | CyberTiel 35B-A3B Q4_K_XL (llama.cpp) |
|---|---|---|
| Solved | **9 / 10** | **7 / 10** |
| Failed | LockTalk | It Has Begun, motp, LockTalk |
| Total wall time | ~14.9 h | ~1.7 h |
| Decode speed | ~15–21 tok/s | ~79 tok/s |

Keep these caveats in mind:

- Ten tasks and a single run each make these numbers noisy.
- The subset leaves out pwn and reversing, and it is easier than the full benchmark, so the results can't be compared to published Cybench scores.
- Each model uses its recommended sampling settings: temperature 0.6 for both, with `top_k` 20 for Qwen and `top_k` 40 plus `min_p` 0.05 for CyberTiel.

## Files

```
~/spark-llm/
├── .api_key            # random bearer key (mode 600)
├── .active             # which model and engine are running, and the address it listens on
├── .active.json        # the running engine's configuration (recorded in --speed results)
├── models/             # downloaded weights, one folder per model and format
├── polystrata/         # PolyStrata checkout, its strata-poly build and the Strata engine (terminal/engine-feed)
├── polystrata-data/    # packs, run configs and the DFlash2 drafter (dflash2/)
├── tensorfold/         # TensorFold release
├── vllm-auth/          # the key-checking middleware mounted into the vLLM container
├── venv/               # Hugging Face CLI
├── speed/              # speed and parity results (JSON)
├── logs/               # native engines' server logs
├── cybench/            # patched Cybench checkout (.env holds the key, mode 600)
└── results/            # per-run Cybench logs and tasks.tsv
```

## Troubleshooting

- **`Access denied ... requires approval` when downloading.** Request access on the model page, then set `HF_TOKEN` or run `hf auth login`.
- **`cannot talk to Docker`.** Add your user to the `docker` group and log in again.
- **`port 8000 is still in use after stopping everything`.** Something outside the script holds the port: `ss -ltnp '( sport = :8000 )'` shows what.
- **`did not refuse requests without a valid key`.** A server answered an unauthenticated request after start, so the script stopped it. The lines above the error name each path and status. Do not work around it: the engine's key handling has to be fixed first.
- **PolyStrata or TensorFold exits at start.** Read `~/spark-llm/logs/polystrata-server.log` (the engine's own log: `~/spark-llm/logs/polystrata-<model>.log`) or `~/spark-llm/logs/tensorfold-<model>.log`. A failed PolyStrata build logs to `~/spark-llm/logs/polystrata-build.log`. For Qwen3.8-Flash-Next, the server log's `[strata] drafting:` line says which drafter the engine started with.
- **`PolyStrata is serving a model: run --stop first` during an install.** PolyStrata's updater does not install the Strata engine while one of its servers runs. Stop the model, install, start again. The native engines start with a reduced environment (see Security notes); a variable they need that is not on the list in `scrub_env` has to be added there.
- **Out-of-memory errors or a slow system.** Lower `CTX` (or `VLLM_GPU_UTIL` for vLLM). Unified memory is shared with the OS.
- **A Cybench task shows `NOT RUN`.** Read `results/<run>/task-N.log`. The usual causes are an x86-only challenge, a failed Docker build, or a leftover container holding a port. The script removes a task's containers after each task.

## Security notes

- By default the API listens on every interface (`BIND=0.0.0.0`), so other machines on your network can reach it. `BIND=127.0.0.1` keeps it to this machine; Cybench then cannot reach it, because its agent container connects over the LAN address.
- Firewalls: ports that Docker publishes (vLLM and llama.cpp) bypass ufw. Docker redirects their traffic in the `nat` table before it reaches the `INPUT` chain ufw uses, so `ufw deny 8000` does not block them. Put such rules in Docker's `DOCKER-USER` chain instead, for example `sudo iptables -I DOCKER-USER -p tcp -m conntrack --ctorigdstport 8000 ! -s 192.168.1.0/24 -j DROP` to allow only your LAN (see Docker's "Packet filtering and firewalls" docs). The native engines (PolyStrata, TensorFold) listen on the host directly, so ufw does cover them.
- Every inference route of every engine needs the key. vLLM's own key check covers only `/v1`, `/v2`, `/inference` and `/cohere`, and its `POST /invocations` would serve the model without one, so the script mounts a small middleware into the vLLM container (`$SPARK_LLM_HOME/vllm-auth/spark_llm_auth.py`, loaded with `--middleware`) that requires the key on every route except `/health` and `/ping`. After every start the script proves it: POSTs without a key and with a wrong key to `/v1/chat/completions` (and, on vLLM, `/invocations`) must get 401 or 403, and other inference paths must refuse too or not exist. Otherwise the script stops the server and exits with an error. Health checks, and llama.cpp's model list (`/v1/models`), answer without a key.
- The API key, and Cybench's `.env` that contains it, are written with mode `600`, are never printed (not even by `--dry-run`), and never appear on a command line: vLLM reads it from `VLLM_API_KEY` in its container's environment, llama.cpp and TensorFold from the key file (`--api-key-file`; read-only mount for llama.cpp), PolyStrata from `STRATA_API_KEY`, and the script's own `curl` checks from standard input (`curl -K -`). Members of the `docker` group can still read it with `docker inspect`, but that group is equivalent to root anyway.
- The script never stores or logs your Hugging Face token. The native engines start with a reduced environment (paths, locale, CUDA/NVIDIA, Python and engine variables), so your token and other credentials exported in the shell do not reach them. PolyStrata's install tools (its updater, drafter fetch and pack tool) run like `pip` does, with your shell's environment.
- What is pinned: PolyStrata's source (a commit of our fork, `PS_REF`); the Strata engine (the signed package of that release's update feed: PolyStrata's updater checks the signature, and every start checks each file against the package's manifest); the DFlash2 drafter (PolyStrata's pinned size and SHA-256; a start refuses a file that does not match); the stock Qwen3.8-Flash-Next files (the Hugging Face revision PolyStrata pins); and TensorFold's release tarball (its SHA-256, `TF_SHA256`, recorded in the script; another `TF_VERSION` needs its own `TF_SHA256`). What is not: the container images are pulled by tag (set `VLLM_IMAGE` / `LLAMA_IMAGE` to `image@sha256:<digest>` to pin them; `docker image inspect --format '{{index .RepoDigests 0}}' <image>` prints the digest you have), the Hugging Face CLI and `ninja` come from PyPI unpinned, PolyStrata's Python requirements and the CUDA wheels by version but without hashes, OrcaRouter's Qwen3.8-Flash-Next files at the repo's current revision, and vLLM runs with `--trust-remote-code` on the model repo.
- Cybench task containers run with `--privileged`, as upstream does. Run benchmarks only on a machine you're comfortable using for that.

## Acknowledgements

- Model authors: [orcarouter](https://huggingface.co/orcarouter) (Qwen3.8-27B Uncensored, Qwen3.8-Flash-Next Uncensored) and [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll) (CyberTiel), building on the Qwen base models; [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) (Qwen3.8-Flash-Next GSQ-RCO GGUF); [z-lab](https://huggingface.co/z-lab) (Qwen3.8-27B DFlash2 drafter) and [incoai](https://huggingface.co/incoai) (Qwen3.8-Flash-Next DFlash2 drafter, tuned by PolyStrata)
- [vLLM](https://github.com/vllm-project/vllm) and the AEON DGX Spark build
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [PolyStrata](https://github.com/simplexai-labs/PolyStrata) and [Strata](https://github.com/Niko1221/Strata)
- [TensorFold](https://github.com/ashhart/TensorFold)
- [Cybench](https://github.com/andyzorigin/cybench)

## License

[MIT](LICENSE). The models, engines, container images and Cybench each have their own licenses.
