#!/usr/bin/env bash
# spark-llm.sh - install, run, benchmark and clean up open-weight models on an NVIDIA DGX Spark.
#
# Models (see `--list`) run on one of four engines, one model at a time, behind an OpenAI-compatible API on :8000
# (bearer key required on every inference route; checked at each start). Each engine reads its own format of the same weights:
#   vllm        vLLM, Spark build (AEON image)               NVFP4 checkpoints
#   llamacpp    llama.cpp server (CUDA 13, arm64)            GGUF
#   polystrata  PolyStrata (r0yfire fork, built for GB10)     GGUF MoE models, natively (no Docker)
#   tensorfold  TensorFold native release (CUDA, sm_121)     MLX 4-bit checkpoints + DFlash2 drafter, natively
# Interactive menu by default. Flags for scripting:
#   --list                                    models and the engines each one runs on
#   --install MODEL [--engine E]              download weights + the engine (image, PolyStrata build or TensorFold release)
#   --start   MODEL [--engine E]              (re)start that model; E defaults to the model's first engine
#   --stop | --status
#   --speed [cases]                           tokens/s of the running model (prefill + decode), see speedtest.py
#   --parity                                  greedy answers to fixed prompts, to compare two engines on one model
#   --cybench-setup                           clone + patch Cybench, pull its image
#   --cybench-run [N]                         Cybench (unguided, 15 iterations) on the running model
#   --cybench-summary <results-dir>           per-task table for a finished run
#   --cleanup [stop|containers|weights|all]
#   --yes (no confirmations) | --dry-run (print, change nothing) | -h
# Hugging Face auth (needed for gated repos): export HF_TOKEN=... or run `hf auth login` once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
BASE="${SPARK_LLM_HOME:-$HOME/spark-llm}"
PORT="${PORT:-8000}"
BIND="${BIND:-0.0.0.0}"   # API listen address: 0.0.0.0 = every interface (LAN clients), 127.0.0.1 = this machine only
VLLM_IMAGE="${VLLM_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
LLAMA_IMAGE="${LLAMA_IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda13}"
CYBENCH_IMAGE="cybench/cybench:latest"
CYBENCH_DIR="$BASE/cybench"
CYBENCH_ITER="${CYBENCH_ITER:-15}"
CYBENCH_MAX_IN="${CYBENCH_MAX_IN:-120000}"
CYBENCH_MAX_OUT="${CYBENCH_MAX_OUT:-32768}"
CYBENCH_TASK_TIMEOUT="${CYBENCH_TASK_TIMEOUT:-21600}"  # seconds per task: a safety net only; the real budget is CYBENCH_ITER iterations
CYBENCH_ONLY="${CYBENCH_ONLY:-}"                        # optional comma-separated task numbers to run, e.g. "6,7,9,10"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.70}"                  # unified memory: stay <= 0.8
# MTP with 4 draft tokens: measured on the Spark against 2 and 6 (Qwen3.8-27B NVFP4: +10% decode over 2, 6 no better)
VLLM_SPEC_DEFAULT='{"method":"mtp","num_speculative_tokens":4}'
VLLM_SPEC="${VLLM_SPEC-$VLLM_SPEC_DEFAULT}"            # speculative decoding for vLLM models (JSON for --speculative-config); VLLM_SPEC= turns it off
CTX="${CTX:-262144}"
# PolyStrata: the r0yfire fork (upstream simplexai-labs/PolyStrata + the aarch64 build fix), pinned.
PS_REPO="${PS_REPO:-https://github.com/r0yfire/PolyStrata.git}"
PS_REF="${PS_REF:-34e454bbefe03ddd0cd449bce85c1fd2f8ba4029}"
PS_DIR="$BASE/polystrata"; PS_DATA="$BASE/polystrata-data"; PS_PID="$BASE/.polystrata.pid"
KEYFILE="$BASE/.api_key"; ACTIVE="$BASE/.active"; ACTIVE_META="$BASE/.active.json"
YES=0; DRY=0

# Model registry, one row per (model, engine): each engine reads its own format of the same weights.
#   key|engine|title|HF repo|--include pattern ("," separates several)|folder under models/|served name
# A model's first row is its default engine: the fastest one measured on the Spark (README, "Results").
MODELS=(
 "qwen27b-abl|tensorfold|Qwen3.8-27B Uncensored, MLX 4-bit + DFlash2|orcarouter/Qwen3.8-27B-Uncensored-MLX|4-bit/*|qwen27b-abl-mlx|Qwen3.8-27B-Uncensored"
 "qwen27b-abl|vllm|Qwen3.8-27B Uncensored, NVFP4 + MTP|orcarouter/Qwen3.8-27B-Uncensored-NVFP4||qwen-nvfp4|Qwen3.8-27B-Uncensored"
 "cybertiel|polystrata|CyberTiel 35B-A3B UD-Q4_K_XL|peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP|*UD-Q4_K_XL.gguf|cybertiel-q4|CyberTiel"
 "cybertiel|llamacpp|CyberTiel 35B-A3B UD-Q4_K_XL|peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP|*UD-Q4_K_XL.gguf|cybertiel-q4|CyberTiel"
 "cybertiel-q8|polystrata|CyberTiel 35B-A3B UD-Q8_K_XL|peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP|*UD-Q8_K_XL.gguf|cybertiel-q8|CyberTiel"
 "cybertiel-q8|llamacpp|CyberTiel 35B-A3B UD-Q8_K_XL|peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP|*UD-Q8_K_XL.gguf|cybertiel-q8|CyberTiel"
)
# DFlash2 drafter for the Qwen3.8-27B family (TensorFold drafts with it; vLLM can too via VLLM_SPEC)
DRAFTER_27B_REPO="${DRAFTER_27B_REPO:-z-lab/Qwen3.8-27B-DFlash2}"
# TensorFold: native release (CUDA kernels embedded), pinned by version AND by the tarball's SHA-256 recorded here.
# The hash is not taken from the release page (release assets can be replaced), so a swapped tarball fails the check.
# Another TF_VERSION needs its own TF_SHA256 (sha256sum of that release's linux-aarch64 tarball, checked out of band).
TF_VERSION="${TF_VERSION:-1.3.6}"; TF_DIR="$BASE/tensorfold"; TF_PID="$BASE/.tensorfold.pid"
TF_SHA256_1_3_6=7f9790480e0a0f4baef73eaf0a5168ffee6d0ddb0e8f11102908f7894d14eb32
TF_SHA256="${TF_SHA256:-}"; if [ -z "$TF_SHA256" ] && [ "$TF_VERSION" = 1.3.6 ]; then TF_SHA256=$TF_SHA256_1_3_6; fi

# Fixed Cybench subset: 10 tasks from the official task_list.txt that run on the aarch64 Spark. Excluded: tasks shipping x86-64
# binaries/services (no amd64 emulation / x86 disassembler) and tasks whose Docker build fails on archived Debian repos (apt 404).
SUBSET=(
 "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic"
 "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Primary Knowledge"
 "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Very Easy] It Has Begun"
 "benchmark/hackthebox/cyber-apocalypse-2024/forensics/[Very Easy] Urgent"
 "benchmark/hackthebox/cyber-apocalypse-2024/web/[Very Easy] Flag Command"
 "benchmark/hkcert-ctf/CTF-Challenges/CTF-2022/22-back-to-the-past"
 "benchmark/project-sekai-ctf/sekaictf-2023/forensics/eval-me"
 "benchmark/hkcert-ctf/CTF-Challenges/CTF-2022/05-motp"
 "benchmark/hackthebox/cyber-apocalypse-2024/web/[Medium] LockTalk"
 "benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Medium] Partial Tenacity"
)

say(){ printf '\033[1m%s\033[0m\n' "$*"; }
die(){ printf 'error: %s\n' "$*" >&2; exit 1; }
# run [-q | -qq] CMD...: run CMD, or under --dry-run only print it. -q drops CMD's stdout, -qq its stdout and stderr.
# The redirections live here, not at the call site, so a dry run still shows the command it skips.
run(){
  local q=""; case "${1:-}" in -q|-qq) q=$1; shift;; esac
  if [ "$DRY" = 1 ]; then printf '+ %s\n' "$*"; return 0; fi
  case "$q" in -q) "$@" >/dev/null;; -qq) "$@" >/dev/null 2>&1;; *) "$@";; esac
}
confirm(){ [ "$YES" = 1 ] && return 0; read -r -p "$1 [y/N] " a; [[ "$a" =~ ^[Yy]$ ]]; }
need(){ command -v "$1" >/dev/null 2>&1 || die "missing: $1"; }

# The bearer key never goes on a command line (ours, curl's, docker's or a server's): servers read it from a file
# or the environment, and curl gets its header from `-K -` fed by the printf builtin.
ensure_key(){   # create the key file once (mode 600); never under --dry-run
  [ "$DRY" = 1 ] && die "internal: ensure_key under --dry-run"
  mkdir -p "$BASE"
  [ -s "$KEYFILE" ] || ( umask 077; python3 -c 'import secrets;print("sk-local-"+secrets.token_hex(16))' > "$KEYFILE" )
}
api_key(){ ensure_key; local k; k=$(<"$KEYFILE"); printf '%s' "${k%%[[:space:]]*}"; }
kcurl(){   # curl with "Authorization: Bearer <key>" taken from the key file, not from argv
  local key; key=$(api_key)
  printf 'header = "Authorization: Bearer %s"\n' "$key" | curl -K - "$@"
}

# ---------------------------------------------------------------- registry
model_rows(){ local r; for r in "${MODELS[@]}"; do [ "${r%%|*}" = "$1" ] && echo "$r"; done; return 0; }
engines_of(){ model_rows "$1" | cut -d'|' -f2 | paste -sd' ' -; }
field(){   # $1 model, $2 engine, $3 column: 3 title, 4 repo, 5 include, 6 folder, 7 served name
  model_rows "$1" | awk -F'|' -v e="$2" -v n="$3" '$2 == e { print $n; exit }'; }
norm(){ local m="${1:-}"
  case "$m" in qwen) m=qwen27b-abl;; tiel|cybertiel-q4) m=cybertiel;; tiel-q8) m=cybertiel-q8;; esac
  [ -n "$(model_rows "$m")" ] || die "unknown model '$m' (see --list)"; echo "$m"; }
engine_for(){   # $1 model, $2 requested engine (may be empty): the engine, checked against the registry
  local engines; engines=$(engines_of "$1")
  [ -z "${2:-}" ] && { echo "${engines%% *}"; return; }
  [[ " $engines " == *" $2 "* ]] || die "$1 does not run on '$2' (it runs on: $engines)"
  echo "$2"
}
mdir(){ echo "$BASE/models/$(field "$1" "$2" 6)"; }       # $1 model, $2 engine
served(){ field "$1" "$2" 7; }
list_models(){
  printf '  %-14s %-11s %s\n' MODEL ENGINE "TITLE   (first engine of a model = its default)"
  local r k e t; for r in "${MODELS[@]}"; do IFS='|' read -r k e t _ <<<"$r"; printf '  %-14s %-11s %s\n' "$k" "$e" "$t"; done
}

hf_cli(){
  need python3
  if [ ! -x "$BASE/venv/bin/hf" ]; then
    say "Installing the Hugging Face CLI into $BASE/venv"
    run python3 -m venv "$BASE/venv"
    run "$BASE/venv/bin/pip" install -q -U "huggingface_hub[cli]"
  fi
}

is_docker_engine(){ case "$1" in vllm|llamacpp) return 0;; *) return 1;; esac; }
need_docker(){
  need docker
  docker info >/dev/null 2>&1 || die "cannot talk to Docker (is $USER in the docker group?)"
}
make_dirs(){ [ "$DRY" = 1 ] || mkdir -p "$BASE/models" "$BASE/logs" "$BASE/results" "$BASE/speed"; }
preflight(){   # $1 engine (optional): before an install. Docker only for the Docker engines; disk space for the download.
  if [ -z "${1:-}" ] || is_docker_engine "$1"; then need_docker; fi
  make_dirs
  local d=$BASE free; while [ ! -d "$d" ]; do d=$(dirname "$d"); done   # a dry run creates nothing: measure the nearest existing parent
  free=$(df -BG --output=avail "$d" | tail -1 | tr -dc 0-9)
  [ "${free:-0}" -ge 80 ] || die "need ~80 GB free in $BASE (have ${free:-?} GB)"
}

download(){   # $1 model, $2 engine
  local repo inc p; repo=$(field "$1" "$2" 4); inc=$(field "$1" "$2" 5)
  [ -n "${HF_TOKEN:-}" ] || [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ] || \
    say "Note: some repos are gated. Request access on Hugging Face, then export HF_TOKEN or run 'hf auth login'."
  say "Downloading $repo${inc:+ ($inc)}"
  # one --include per pattern: `hf download` reads a second pattern after one flag as a file name
  local args=(); IFS=',' read -ra pats <<<"$inc"; for p in "${pats[@]}"; do [ -n "$p" ] && args+=(--include "$p"); done
  run "$BASE/venv/bin/hf" download "$repo" "${args[@]}" --local-dir "$(mdir "$1" "$2")"
}

drafter_27b(){ echo "$BASE/models/qwen27b-dflash2"; }
needs_drafter(){ [[ "$1" == qwen27b* ]] && { [ "$2" = tensorfold ] || [[ "$2" = vllm && "$VLLM_SPEC" == *dflash* ]]; }; }

install_model(){   # $1 model, $2 engine (optional)
  local m e; m=$(norm "$1"); e=$(engine_for "$m" "${2:-}"); preflight "$e"; hf_cli
  download "$m" "$e"
  if needs_drafter "$m" "$e"; then
    say "Downloading the DFlash2 drafter ($DRAFTER_27B_REPO)"
    run "$BASE/venv/bin/hf" download "$DRAFTER_27B_REPO" --local-dir "$(drafter_27b)"
  fi
  case "$e" in
    vllm) run docker pull "$VLLM_IMAGE" ;;
    llamacpp) run docker pull "$LLAMA_IMAGE" ;;
    polystrata) ps_install ;;
    tensorfold) tf_install ;;
    *) die "engine '$e' is not wired into install_model" ;;
  esac
  say "Installed $m ($e)"
}

# ---------------------------------------------------------------- PolyStrata (native, not Docker)
# strata-poly has no aarch64 release package, so it is built here for the GB10 (sm_121) from the pinned fork.
ps_install(){
  need git; need cmake; need python3
  local nvcc=/usr/local/cuda/bin/nvcc; [ -x "$nvcc" ] || nvcc=$(command -v nvcc) || die "the CUDA 13 toolkit (nvcc) is needed"
  if [ ! -d "$PS_DIR/.git" ]; then say "Cloning PolyStrata ($PS_REPO)"; run git clone -q "$PS_REPO" "$PS_DIR"; fi
  run git -C "$PS_DIR" fetch -q origin "$PS_REF" 2>/dev/null || true
  run git -C "$PS_DIR" checkout -q "$PS_REF"
  if [ ! -x "$PS_DIR/terminal/.venv/bin/python" ]; then
    say "Creating PolyStrata's Python environment"
    run python3 -m venv "$PS_DIR/terminal/.venv"
  fi
  run "$PS_DIR/terminal/.venv/bin/pip" install -q --disable-pip-version-check -r "$PS_DIR/terminal/requirements.txt" ninja
  [ "$DRY" = 1 ] && return
  local stamp="$PS_DIR/build-poly-spark/.built-from"
  if [ -x "$PS_DIR/build-poly-spark/strata-poly" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$PS_REF" ]; then
    say "strata-poly already built from $PS_REF"; return; fi
  say "Building strata-poly for sm_121 (about 10 minutes)"
  ( cd "$PS_DIR" && PATH="$PS_DIR/terminal/.venv/bin:$(dirname "$nvcc"):$PATH" &&
    cmake -G Ninja -S engine -B build-poly-spark -DCMAKE_BUILD_TYPE=Release -DGGML_CCACHE=OFF \
      -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=OFF \
      "-DCMAKE_CUDA_ARCHITECTURES=121-real;121-virtual" "-DCMAKE_CUDA_COMPILER=$nvcc" >"$BASE/logs/polystrata-build.log" 2>&1 &&
    # CUDA 13.0's ptxas now and then fails on a PTX file it just wrote; a second pass gets past it (PolyStrata issue #45)
    { cmake --build build-poly-spark --target strata-poly -j 16 >>"$BASE/logs/polystrata-build.log" 2>&1 ||
      cmake --build build-poly-spark --target strata-poly -j 16 >>"$BASE/logs/polystrata-build.log" 2>&1; } ) ||
    die "strata-poly build failed: see $BASE/logs/polystrata-build.log"
  echo "$PS_REF" > "$stamp"
}

first_gguf(){ [ -d "$1" ] || return 0; find "$1" -name '*.gguf' | sort | head -1; }   # $1 dir; empty when none (or no dir)

ps_config(){   # $1 model -> prints ONLY the path of its PolyStrata run config (tokenizer pack + JSON); messages go to stderr
  local m=$1 g pack cfg; g=$(first_gguf "$(mdir "$m" polystrata)")
  [ -n "$g" ] || die "no GGUF under $(mdir "$m" polystrata)"
  pack="$PS_DATA/packs/$m"; cfg="$PS_DATA/strata-$m.json"; mkdir -p "$pack"
  if [ ! -s "$pack/tokenizer/vocab.json" ]; then
    say "Exporting $m's tokenizer for PolyStrata" >&2
    ( cd "$PS_DIR/terminal" && PYTHONPATH=../engine/third_party/strata-gguf-py \
      .venv/bin/python tools/strata_tokenizer.py --gguf "$g" --out "$pack" >/dev/null ) || die "tokenizer export failed"
  fi
  # No speculative-decoding flag: strata-poly drafts with the GGUF's own MTP head on its own (its log reports
  # "drafts accepted N of M"). The expert profile need not exist yet: strata-poly creates it on the first start.
  python3 - "$cfg" "$PS_DIR" "$g" "$pack" "$CTX" "$(served "$m" polystrata)" "$BASE/logs/polystrata-$m.log" <<'PY'
import json, sys
cfg, ps, gguf, pack, ctx, name, log = sys.argv[1:]
json.dump({"exe": f"{ps}/build-poly-spark/strata-poly",
           "args": ["--model", gguf, "--max-context", ctx, "--expert-profile", f"{pack}/expert-profile.txt",
                    "--prompt-batch", "2048"],
           "cwd": f"{ps}/terminal", "tokenizer": f"{pack}/tokenizer", "model_name": name, "log": log},
          open(cfg, "w"), indent=1)
PY
  echo "$cfg"
}

ps_start(){   # $1 model, $2 run config from ps_config
  local m=$1 cfg=${2:-} key
  [ "$DRY" = 1 ] && { say "(dry run: would start PolyStrata for $m on $BIND:$PORT)"; return; }
  say "Starting $(field "$m" polystrata 3) on PolyStrata (strata-poly, ${CTX} ctx)"
  key=$(api_key)
  # The key goes in through the environment: serve/server.py's --api-key defaults to $STRATA_API_KEY. Not argv.
  ( cd "$PS_DIR/terminal" || die "missing $PS_DIR/terminal (run --install $m --engine polystrata)"
    STRATA_API_KEY=$key native_launch "$PS_PID" "$BASE/logs/polystrata-server.log" \
      .venv/bin/python serve/server.py --engine strata --config "$cfg" --host "$BIND" --port "$PORT" )
}

# ---------------------------------------------------------------- native engines (PolyStrata, TensorFold)
# A native server runs in its own session, so --stop can end it and every process it spawned together.
# The PID file holds "PID START": the session leader's PID (`exec` makes it the server itself, not a wrapper shell)
# and its start time from /proc. Both must match, and the command line must contain the marker below, before
# anything is signaled: a PID file that outlived its server (a reboot, a crash) must not get an unrelated process killed.
PS_MARK="$PS_DATA/strata-"; TF_MARK="$TF_DIR/"   # in the server's argv: its config path / its executable path

# Native servers get only this environment, so credentials exported in the calling shell (HF_TOKEN, cloud keys,
# OPENAI_API_KEY...) do not reach third-party code. Unset in a subshell: no value ever goes through an argv.
scrub_env(){
  local v
  for v in $(compgen -e); do
    case "$v" in
      PATH|HOME|USER|LOGNAME|SHELL|LANG|LANGUAGE|TERM|TZ|TMPDIR|LD_LIBRARY_PATH|OMP_NUM_THREADS) ;;
      LC_*|XDG_*|CUDA_*|NVIDIA_*|NCCL_*|CUBLAS_*|PYTORCH_*|TORCH_*|TRITON_*|PYTHON*|STRATA_*|TENSORFOLD_*) ;;
      *) unset "$v" 2>/dev/null || true ;;
    esac
  done
}

native_launch(){   # $1 pid file, $2 log, $3.. command
  local pidf=$1 log=$2; shift 2
  rm -f "$pidf"
  ( scrub_env
    exec setsid bash -c 'read -r s < "/proc/$$/stat"; s=${s##*) }; read -ra f <<<"$s"; echo "$$ ${f[19]}" > "$0"; exec "$@"' \
      "$pidf" "$@" >"$log" 2>&1 </dev/null ) &
  for _ in 1 2 3 4 5; do [ -s "$pidf" ] && break; sleep 1; done
}

proc_start(){ local s f; s=$(<"/proc/$1/stat") || return 1; s=${s##*) }; read -ra f <<<"$s"; echo "${f[19]}"; }   # $1 pid

native_pid(){   # $1 pid file, $2 marker -> prints the PID when the file names a server of ours that is still running
  local pid="" st="" cur cmd
  [ -s "$1" ] || return 1
  read -r pid st _ < "$1" || true
  [[ "$pid" =~ ^[0-9]+$ ]] && [ -r "/proc/$pid/stat" ] || return 1
  cur=$(proc_start "$pid" 2>/dev/null) || return 1
  [ -z "$st" ] || [ "$st" = "$cur" ] || return 1          # older PID files hold no start time: the marker decides alone
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
  [[ "$cmd" == *"$2"* ]] || return 1
  echo "$pid"
}
native_alive(){ native_pid "$1" "$2" >/dev/null; }

session_gone(){ local i; for ((i = 0; i < $2; i++)); do pgrep -s "$1" >/dev/null || return 0; sleep 1; done; ! pgrep -s "$1" >/dev/null; }

native_stop(){   # $1 pid file, $2 label, $3 marker
  [ -s "$1" ] || return 0
  local pid old="" left=""
  if ! pid=$(native_pid "$1" "$3"); then
    read -r old _ < "$1" || true
    # The leader is gone but its session lives on: its PID cannot have been reused while the session exists,
    # so these are most likely the server's own children. Report them; do not guess and kill.
    if [[ "$old" =~ ^[0-9]+$ ]] && [ ! -e "/proc/$old" ]; then left=$(pgrep -s "$old" | paste -sd' ' - || true); fi
    [ -z "$left" ] || say "$2: its server is gone, but processes of its session $old are still running ($left). Check: ps -o pid,args -s $old"
    if [ "$DRY" = 1 ]; then say "$2: not running (dry run: would remove the stale PID file $1)"; return 0; fi
    say "$2: not running (removing the stale PID file $1)"; rm -f "$1"; return 0
  fi
  if [ "$DRY" = 1 ]; then say "(dry run: would stop $2, pid $pid, and every process in its session)"; return 0; fi
  say "Stopping $2"
  pkill -TERM -s "$pid" || true
  if ! session_gone "$pid" 60; then
    say "$2 did not exit within 60 s; sending SIGKILL"
    pkill -KILL -s "$pid" || true
    session_gone "$pid" 15 || die "$2 (session $pid) is still running after SIGKILL: ps -o pid,args -s $pid"
  fi
  rm -f "$1"
}

tf_bin(){ echo "$TF_DIR/tensorfold-$TF_VERSION-linux-aarch64/bin/tensorfold-native"; }

tf_install(){
  [ -x "$(tf_bin)" ] && { say "TensorFold $TF_VERSION already installed"; return; }
  need curl; need sha256sum
  [[ "$TF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "no pinned SHA-256 for TensorFold $TF_VERSION: set TF_SHA256 to the tarball's sha256sum"
  local a="tensorfold-$TF_VERSION-linux-aarch64.tar.gz" u="https://github.com/ashhart/TensorFold/releases/download/v$TF_VERSION"
  say "Installing TensorFold $TF_VERSION (native, CUDA kernels embedded)"
  if [ "$DRY" = 1 ]; then
    run mkdir -p "$TF_DIR"; run curl -fsSL -o "$TF_DIR/$a" "$u/$a"
    printf '+ %s\n' "check sha256 of $TF_DIR/$a is $TF_SHA256" "tar -xzf $TF_DIR/$a -C $TF_DIR"
    return
  fi
  mkdir -p "$TF_DIR"
  curl -fsSL -o "$TF_DIR/$a" "$u/$a" || die "TensorFold download failed ($u/$a)"
  # The expected hash is the one pinned above, never the release's own .sha256; `-c -` hashes exactly this file.
  ( cd "$TF_DIR" && printf '%s  %s\n' "$TF_SHA256" "$a" | sha256sum -c --quiet - ) ||
    { rm -f "$TF_DIR/$a"; die "TensorFold $TF_VERSION tarball does not match the pinned SHA-256 ($TF_SHA256); deleted it"; }
  tar -xzf "$TF_DIR/$a" -C "$TF_DIR" || die "could not unpack $TF_DIR/$a"
  [ -x "$(tf_bin)" ] || die "the TensorFold $TF_VERSION tarball has no $(tf_bin)"
}

tf_model_path(){   # $1 model: a pattern like "4-bit/*" means the model is that subfolder
  local dir inc; dir=$(mdir "$1" tensorfold); inc=$(field "$1" tensorfold 5)
  if [[ "$inc" == */* ]]; then echo "$dir/${inc%%/*}"; else echo "$dir"; fi
}

tf_start(){   # $1 model
  local m=$1 path drafter=()
  path=$(tf_model_path "$m")
  if needs_drafter "$m" tensorfold; then drafter=(--drafter "$(drafter_27b)"); fi
  [ "$DRY" = 1 ] && { say "(dry run: would serve $path on TensorFold, $BIND:$PORT)"; return; }
  say "Starting $(field "$m" tensorfold 3) on TensorFold $TF_VERSION (${CTX} ctx)"
  # shellcheck disable=SC2086   # TF_EXTRA is a list of extra arguments
  native_launch "$TF_PID" "$BASE/logs/tensorfold-$m.log" "$(tf_bin)" serve "$path" "${drafter[@]}" \
    --context "$CTX" --parallel 1 --host "$BIND" --port "$PORT" --api-key-file "$KEYFILE" \
    --name "$(served "$m" tensorfold)" --no-update-check ${TF_EXTRA:-}
}

# ---------------------------------------------------------------- vLLM key check for every route
# vLLM's own key check (VLLM_API_KEY / --api-key) guards only /v1, /v2, /inference and /cohere: the SageMaker-style
# POST /invocations serves the same inference without a key. This module, mounted read-only into the container and
# loaded with --middleware, requires the key on every route except the health probes.
vllm_auth_module(){
  mkdir -p "$BASE/vllm-auth"
  cat > "$BASE/vllm-auth/spark_llm_auth.py" <<'PY'
"""Bearer-key gate in front of every vLLM route (written by spark-llm.sh, loaded with --middleware).

vLLM's own key check covers only /v1, /v2, /inference and /cohere; POST /invocations (the SageMaker
route) serves inference without it. This ASGI middleware requires "Authorization: Bearer $VLLM_API_KEY"
on every HTTP and WebSocket request except the health probes /health and /ping, and CORS preflights
(OPTIONS, which reach no handler). It refuses to load without a key.
"""
import hmac
import os

OPEN_PATHS = frozenset({"/health", "/ping"})


class RequireApiKey:
    def __init__(self, app):
        key = os.environ.get("VLLM_API_KEY", "")
        if not key:
            raise RuntimeError("spark_llm_auth: VLLM_API_KEY is empty; refusing to serve without a key")
        self.app = app
        self.key = key.encode()

    def _authorized(self, scope):
        for name, value in scope.get("headers") or ():
            if name.lower() == b"authorization":
                scheme, _, token = value.strip().partition(b" ")
                return scheme.lower() == b"bearer" and hmac.compare_digest(token.strip(), self.key)
        return False

    async def __call__(self, scope, receive, send):
        kind = scope.get("type")
        if kind not in ("http", "websocket") or (kind == "http" and scope.get("method") == "OPTIONS"):
            return await self.app(scope, receive, send)
        path, root = scope.get("path") or "", scope.get("root_path") or ""
        if root and path.startswith(root):
            path = path[len(root):] or "/"
        if path in OPEN_PATHS or self._authorized(scope):
            return await self.app(scope, receive, send)
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = b'{"error":"Unauthorized"}'
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": body})
PY
}
image_pythonpath(){   # $1 image -> the PYTHONPATH it sets (kept behind ours)
  { docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$1" 2>/dev/null || true; } | sed -n 's/^PYTHONPATH=//p' | head -1
}

# ---------------------------------------------------------------- start / stop
set_model_args(){   # $1 engine, $2 model -> MODEL_ARGS: the flags that depend on the model, not only on the engine
  case "$1:$2" in
    vllm:qwen27b*) MODEL_ARGS=(--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder) ;;
    # CyberTiel's GGUF carries an MTP draft head; its recommended sampling
    llamacpp:cybertiel*) MODEL_ARGS=(--spec-type draft-mtp --temp 0.6 --top-p 0.95 --top-k 40 --min-p 0.05) ;;
    polystrata:*|tensorfold:*) MODEL_ARGS=() ;;
    *) die "no $1 settings for model '$2': add them to set_model_args" ;;
  esac
}

missing_for(){   # $1 model, $2 engine -> prints what a start needs but is not there, one item per line (nothing = runnable)
  local m=$1 e=$2 d; d=$(mdir "$m" "$e")
  case "$e" in
    vllm)
      [ -s "$d/config.json" ] || echo "weights ($d)"
      docker image inspect "$VLLM_IMAGE" >/dev/null 2>&1 || echo "image $VLLM_IMAGE"
      if [[ "$VLLM_SPEC" == *dflash* ]] && [ ! -s "$(drafter_27b)/config.json" ]; then echo "DFlash2 drafter ($(drafter_27b))"; fi ;;
    llamacpp)
      [ -n "$(first_gguf "$d")" ] || echo "GGUF weights ($d)"
      docker image inspect "$LLAMA_IMAGE" >/dev/null 2>&1 || echo "image $LLAMA_IMAGE" ;;
    polystrata)
      [ -n "$(first_gguf "$d")" ] || echo "GGUF weights ($d)"
      [ -x "$PS_DIR/build-poly-spark/strata-poly" ] || echo "the strata-poly build"
      [ -x "$PS_DIR/terminal/.venv/bin/python" ] || echo "PolyStrata's Python environment" ;;
    tensorfold)
      [ -s "$(tf_model_path "$m")/config.json" ] || echo "weights ($(tf_model_path "$m"))"
      [ -x "$(tf_bin)" ] || echo "TensorFold $TF_VERSION"
      if needs_drafter "$m" tensorfold && [ ! -s "$(drafter_27b)/config.json" ]; then echo "DFlash2 drafter ($(drafter_27b))"; fi ;;
    *) echo "engine '$e' (not wired into this script)" ;;
  esac
}

check_runnable(){   # $1 model, $2 engine: dies if a start could not work. Runs BEFORE the serving model is stopped.
  local miss; miss=$(missing_for "$1" "$2")
  set_model_args "$2" "$1"
  [ -z "$miss" ] && return 0
  miss=$(paste -sd';' - <<<"$miss")
  if [ "$DRY" = 1 ]; then say "(dry run: $1 on $2 is not installed yet; missing: $miss)"; return 0; fi
  die "cannot start $1 on $2, missing: $miss (run --install $1 --engine $2)"
}

check_bind(){
  [[ "$BIND" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || die "BIND must be an IPv4 address: 0.0.0.0 (all interfaces) or 127.0.0.1 (this machine only)"
}
probe_host(){ if [ "$1" = 0.0.0.0 ]; then echo 127.0.0.1; else echo "$1"; fi; }   # $1 bind address -> where to reach it from here
public_host(){ if [ "$1" = 0.0.0.0 ]; then hostname -I | awk '{print $1}'; else echo "$1"; fi; }

chat_body(){ printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' "$1"; }   # $1 served name
# for the unauthenticated probes: one body that is valid on every inference path probed, so an open path shows up as 200
probe_body(){ printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"prompt":"ping","input":"ping","max_tokens":1,"n_predict":1}' "$1"; }
noauth_status(){   # $1 none|wrong, $2 URL, $3 JSON body -> HTTP status of a POST without a valid key (000 = no answer)
  local h=(); if [ "$1" = wrong ]; then h=(-H "Authorization: Bearer spark-llm-probe-wrong-key"); fi
  curl -s -o /dev/null -w '%{http_code}' --max-time 60 -X POST -H 'Content-Type: application/json' "${h[@]}" \
    --data-raw "$3" "$2" || true
}

# Proves the key is enforced: POSTs without a key and with a wrong one must get 401/403 on the chat endpoint (and, on
# vLLM, on /invocations). Other inference paths must refuse too, or not exist (404/405). Anything else: stop and die.
auth_check(){   # $1 engine, $2 model, $3 base URL
  local e=$1 url=$3 body p a code must=(/v1/chat/completions) may=(/v1/completions /v1/responses /v1/messages /v1/embeddings /completion /completions) bad=()
  if [ "$e" = vllm ]; then must+=(/invocations); else may+=(/invocations); fi
  body=$(probe_body "$(served "$2" "$e")")
  for p in "${must[@]}" "${may[@]}"; do
    for a in none wrong; do
      code=$(noauth_status "$a" "$url$p" "$body")
      case "$code" in 401|403) continue ;; esac
      if [[ " ${may[*]} " == *" $p "* ]] && { [ "$code" = 404 ] || [ "$code" = 405 ]; }; then continue; fi
      bad+=("POST $p with $a key -> HTTP $code")
    done
  done
  if [ ${#bad[@]} -gt 0 ]; then
    printf '  %s\n' "${bad[@]}" >&2
    printf 'error: %s\n' "the $e server did not refuse requests without a valid key (expected 401/403); stopping it" >&2
    stop_all
    die "stopped the $e server: it served requests without a valid key"
  fi
  say "Auth check passed: requests without a valid key are refused (${must[*]} and other inference paths)"
}

wait_ready(){   # $1 engine, $2 model, $3 container (docker engines)
  local e=$1 m=$2 c=$3 url name deadline=$((SECONDS + 1800))
  url="http://$(probe_host "$BIND"):$PORT"; name=$(served "$m" "$e")
  say "Waiting for the server (first start can take 10+ minutes: weights load, kernels compile)"
  while [ "$SECONDS" -lt "$deadline" ]; do
    case "$e" in
      polystrata) native_alive "$PS_PID" "$PS_MARK" || { tail -40 "$BASE/logs/polystrata-server.log"; die "PolyStrata exited"; } ;;
      tensorfold) native_alive "$TF_PID" "$TF_MARK" || { tail -40 "$BASE/logs/tensorfold-$m.log"; die "TensorFold exited"; } ;;
      vllm|llamacpp) [ "$(docker inspect --type container -f '{{.State.Running}}' "$c" 2>/dev/null)" = true ] || { docker logs --tail 60 "$c" 2>&1 || true; die "container $c exited"; } ;;
      *) die "engine '$e' is not wired into wait_ready" ;;
    esac
    # Listening first (the models list), then one real 1-token completion: a front-end server can list models
    # before its engine has loaded.
    if kcurl -fs --max-time 5 -o /dev/null "$url/v1/models" 2>/dev/null &&
       kcurl -fs --max-time 600 -o /dev/null -H 'Content-Type: application/json' \
         --data-raw "$(chat_body "$name")" "$url/v1/chat/completions" 2>/dev/null; then
      auth_check "$e" "$m" "$url"
      say "Ready: http://$(public_host "$BIND"):$PORT/v1  (model: $name, engine: $e)"; return 0
    fi
    sleep 10
  done
  die "timed out waiting for $m on $e"
}

legacy_names(){   # containers older versions of this script started without the spark-llm label
  echo llm-qwen; printf '%s\n' "${MODELS[@]}" | cut -d'|' -f1 | sort -u | sed 's/^/llm-/'
}
our_containers(){   # names of the model containers this script started, and only those
  local all; all=$(docker ps -a --format '{{.Names}}' 2>/dev/null) || return 0
  { docker ps -a --filter label=spark-llm=1 --format '{{.Names}}' 2>/dev/null || true
    grep -Fxf <(legacy_names) <<<"$all" || true; } | sort -u
}

stop_all(){
  local c
  if command -v docker >/dev/null 2>&1; then
    for c in $(our_containers); do say "Stopping $c"; run -q docker rm -f "$c"; done
  fi
  native_stop "$PS_PID" PolyStrata "$PS_MARK"
  native_stop "$TF_PID" TensorFold "$TF_MARK"
  [ "$DRY" = 1 ] || rm -f "$ACTIVE" "$ACTIVE_META"
  # the port must be free before the next start: a server still holding it would answer in the new one's place
  local t=0; while [ "$DRY" = 0 ] && ss -ltn "( sport = :$PORT )" | grep -q LISTEN; do
    [ $t -ge 60 ] && die "port $PORT is still in use after stopping everything (ss -ltnp to see by what)"; sleep 1; t=$((t+1)); done
}

vllm_start(){   # $1 model, $2 container name
  local m=$1 name=$2 spec=$VLLM_SPEC drafter=() specargs=() pp
  if [ -n "$spec" ]; then specargs=(--speculative-config "$spec"); fi
  if [[ "$spec" == *dflash* ]]; then drafter=(-v "$(drafter_27b):/drafter:ro"); fi
  pp=$(image_pythonpath "$VLLM_IMAGE")
  say "Starting $(field "$m" vllm 3) on vLLM (speculative: ${spec:-off}, ${CTX} ctx, gpu-util $VLLM_GPU_UTIL)"
  # The key reaches vLLM as VLLM_API_KEY: `-e VLLM_API_KEY` (no value) copies it from this subshell's environment.
  # shellcheck disable=SC2086   # VLLM_EXTRA is a list of extra arguments
  ( if [ "$DRY" = 0 ]; then VLLM_API_KEY=$(api_key); export VLLM_API_KEY; fi
    run docker run -d --name "$name" --label spark-llm=1 --gpus all --ipc=host -p "$BIND:$PORT:8000" \
      -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
      -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
      -e VLLM_API_KEY -e "PYTHONPATH=/opt/spark-llm-auth${pp:+:$pp}" -v "$BASE/vllm-auth:/opt/spark-llm-auth:ro" \
      -v "$(mdir "$m" vllm):/model:ro" "${drafter[@]}" --entrypoint vllm "$VLLM_IMAGE" \
      serve /model --served-model-name "$(served "$m" vllm)" "${specargs[@]}" \
      --gpu-memory-utilization "$VLLM_GPU_UTIL" --max-model-len "$CTX" --max-num-seqs 4 \
      --trust-remote-code "${MODEL_ARGS[@]}" --middleware spark_llm_auth.RequireApiKey \
      --host 0.0.0.0 --port 8000 ${VLLM_EXTRA:-${QWEN_EXTRA:-}} )
}

llama_start(){   # $1 model, $2 container name
  local m=$1 name=$2 d f; d=$(mdir "$m" llamacpp); f=$(first_gguf "$d")
  say "Starting $(field "$m" llamacpp 3) ($(basename "${f:-gguf}")) on llama.cpp (MTP, ${CTX} ctx)"
  # The key file is mounted read-only and read by --api-key-file: the key is not in any argv.
  # shellcheck disable=SC2086   # LLAMA_EXTRA is a list of extra arguments
  run docker run -d --name "$name" --label spark-llm=1 --gpus all -p "$BIND:$PORT:8000" \
    --health-cmd "curl -fs http://localhost:8000/health || exit 1" --health-interval 30s --health-start-period 10m \
    -v "$d:/models:ro" -v "$KEYFILE:/run/secrets/spark-llm-api-key:ro" "$LLAMA_IMAGE" \
    -m "/models/${f#"$d"/}" --alias "$(served "$m" llamacpp)" -ngl 99 --jinja -fa on \
    --ctx-size "$CTX" -np 1 "${MODEL_ARGS[@]}" \
    --host 0.0.0.0 --port 8000 --api-key-file /run/secrets/spark-llm-api-key ${LLAMA_EXTRA:-${TIEL_EXTRA:-}}
}

start_model(){   # $1 model, $2 engine (optional)
  local m e cfg=""; m=$(norm "$1"); e=$(engine_for "$m" "${2:-}")
  check_bind
  if is_docker_engine "$e"; then need_docker; fi
  check_runnable "$m" "$e"   # before stop_all: a start that cannot work must not take down the model that is serving
  make_dirs
  if [ "$DRY" = 0 ]; then
    ensure_key
    case "$e" in polystrata) cfg=$(ps_config "$m") ;; vllm) vllm_auth_module ;; esac
  fi
  stop_all
  local name="llm-$m"
  case "$e" in
    vllm) vllm_start "$m" "$name" ;;
    llamacpp) llama_start "$m" "$name" ;;
    polystrata) ps_start "$m" "$cfg" ;;
    tensorfold) tf_start "$m" ;;
    *) die "engine '$e' is not wired into start_model" ;;
  esac
  [ "$DRY" = 1 ] && return
  wait_ready "$e" "$m" "$name"   # includes the auth check
  write_active "$m" "$e" "$name"
}

write_active(){   # $1 model, $2 engine, $3 container: .active ("model engine bind") + .active.json (what --speed records)
  local m=$1 e=$2 kv=()
  echo "$m $e $BIND" > "$ACTIVE"
  kv=("model=$m" "engine=$e" "served=$(served "$m" "$e")" "ctx=$CTX" "bind=$BIND" "started=$(date +%Y-%m-%dT%H:%M:%S%z)")
  case "$e" in
    vllm) kv+=("image=$VLLM_IMAGE" "image_id=$(docker image inspect -f '{{.Id}}' "$VLLM_IMAGE" 2>/dev/null || true)"
               "vllm_spec=$VLLM_SPEC" "gpu_util=$VLLM_GPU_UTIL" "extra=${VLLM_EXTRA:-${QWEN_EXTRA:-}}"
               "vllm_version=$(kcurl -fs --max-time 5 "http://$(probe_host "$BIND"):$PORT/version" 2>/dev/null || true)") ;;
    llamacpp) kv+=("image=$LLAMA_IMAGE" "image_id=$(docker image inspect -f '{{.Id}}' "$LLAMA_IMAGE" 2>/dev/null || true)"
                   "gguf=$(basename "$(first_gguf "$(mdir "$m" llamacpp)")")" "extra=${LLAMA_EXTRA:-${TIEL_EXTRA:-}}") ;;
    polystrata) kv+=("ps_repo=$PS_REPO" "ps_ref=$PS_REF" "ps_head=$(git -C "$PS_DIR" rev-parse HEAD 2>/dev/null || true)"
                     "gguf=$(basename "$(first_gguf "$(mdir "$m" polystrata)")")") ;;
    tensorfold) kv+=("tf_version=$TF_VERSION" "tf_sha256=$TF_SHA256" "extra=${TF_EXTRA:-}")
                if needs_drafter "$m" tensorfold; then kv+=("drafter=$DRAFTER_27B_REPO"); fi ;;
  esac
  python3 - "$ACTIVE_META" "${kv[@]}" <<'PY'
import json, sys
meta = dict(a.split("=", 1) for a in sys.argv[2:])
try:
    meta["vllm_version"] = json.loads(meta["vllm_version"])["version"]
except (KeyError, ValueError, TypeError):
    pass
with open(sys.argv[1], "w") as f:
    json.dump(meta, f, indent=1, sort_keys=True)
PY
}

ACTIVE_M=""; ACTIVE_E=""; ACTIVE_B=""
read_active(){   # sets ACTIVE_M / ACTIVE_E / ACTIVE_B from .active; dies when no model is running or the file is not ours
  [ -s "$ACTIVE" ] || die "no model is running (use --start)"
  read -r ACTIVE_M ACTIVE_E ACTIVE_B _ < "$ACTIVE" || true
  [ -n "$(model_rows "$ACTIVE_M")" ] && [[ " $(engines_of "$ACTIVE_M") " == *" $ACTIVE_E "* ]] ||
    die "$ACTIVE says '$(cat "$ACTIVE")', which this version cannot read (written by an older version?): run --start again"
  ACTIVE_B=${ACTIVE_B:-0.0.0.0}
}

native_status(){ if native_alive "$1" "$3"; then say "$2: running (pid $(native_pid "$1" "$3"))"; else say "$2: not running"; fi; }

status(){
  say "Containers"
  if command -v docker >/dev/null 2>&1; then
    local c n=0; for c in $(our_containers); do docker ps -a --filter "name=^/?$c\$" --format '  {{.Names}}  {{.Status}}  {{.Image}}'; n=$((n+1)); done
    [ "$n" -gt 0 ] || echo "  none"
  else echo "  (docker is not installed)"; fi
  native_status "$PS_PID" PolyStrata "$PS_MARK"; native_status "$TF_PID" TensorFold "$TF_MARK"
  say "Active: $(cat "$ACTIVE" 2>/dev/null || echo none)"
  say "Memory"; free -g | sed -n 1,2p
  say "Disk"
  local p paths=(); for p in "$BASE/models"/* "$PS_DIR" "$PS_DATA" "$TF_DIR"; do [ -e "$p" ] && paths+=("$p"); done
  if [ ${#paths[@]} -gt 0 ]; then du -sh "${paths[@]}" 2>/dev/null || true; fi
  [ -n "$(compgen -G "$BASE/models/*")" ] || echo "  no models installed"
  echo "API key file: $KEYFILE"
}

# ---------------------------------------------------------------- speed
speed(){   # $1 mode: speed | parity; $2 cases
  read_active
  local m=$ACTIVE_M e=$ACTIVE_E url out meta='{}' common
  url="http://$(probe_host "$ACTIVE_B"):$PORT/v1"; out="$BASE/speed/$m-$e-$(date +%Y%m%d-%H%M%S)"
  if [ -s "$ACTIVE_META" ]; then meta=$(<"$ACTIVE_META"); fi   # the engine config the server was started with
  # Thinking off, so every engine writes answers, not reasoning, and they compare. speedtest.py checks that it held:
  # a run that produced reasoning (streamed or hidden) is flagged and left out of the medians.
  local eb='{"chat_template_kwargs":{"enable_thinking":false}}'
  local st="$SCRIPT_DIR/speedtest.py"; [ -f "$st" ] || die "speedtest.py is missing beside $0"
  common=(--base-url "$url" --key-file "$KEYFILE" --label "$m-$e" --extra-body "$eb" --meta "$meta")
  make_dirs
  if [ "$1" = parity ]; then
    run python3 "$st" "${common[@]}" --parity --out "$out-parity.json"
  else
    run python3 "$st" "${common[@]}" --cases "${2:-short,4k,16k,32k}" --runs "${SPEED_RUNS:-3}" --out "$out.json"
    if [ "${SPEED_NEEDLE:-0}" -gt 0 ]; then
      run python3 "$st" "${common[@]}" --needle "$SPEED_NEEDLE" --out "$out-needle.json"; fi
  fi
  [ "$DRY" = 1 ] || say "Results: $out*.json"
}

# ---------------------------------------------------------------- Cybench
cybench_setup(){
  preflight; need git
  if [ ! -d "$CYBENCH_DIR/.git" ]; then
    say "Cloning Cybench"; run git clone --depth 1 https://github.com/andyzorigin/cybench.git "$CYBENCH_DIR"
  fi
  [ "$DRY" = 1 ] && { say "(dry run: skipping patch and image pull)"; return; }
  say "Patching Cybench for a local OpenAI-compatible endpoint (base URL, sampling, tokenizer, Linux networking)"
  python3 - "$CYBENCH_DIR" <<'PY'
import sys, re, pathlib
root = pathlib.Path(sys.argv[1])
def patch(path, subs, once=True):
    p = root / path; s = p.read_text()
    if "SPARK_PATCHED" in s: return
    for old, new in subs:
        if old not in s: sys.exit(f"patch failed: {path}: pattern not found: {old[:60]!r}")
        s = s.replace(old, new) if not once else s.replace(old, new, 1)
    p.write_text("# SPARK_PATCHED\n" + s if path.endswith(".py") else s.replace("#!/bin/bash", "#!/bin/bash\n# SPARK_PATCHED", 1))

patch("agent/models/non_helm.py", [
 ("return OpenAI(api_key=self._api_key())",
  'return OpenAI(api_key=self._api_key(), base_url=os.environ.get("OPENAI_BASE_URL") or None, timeout=3600)'),
 ("            temperature=temperature,\n            max_tokens=max_tokens,\n            stop=stop_sequences,\n        )",
  '            temperature=float(os.environ.get("CYBENCH_TEMPERATURE", temperature)),\n'
  '            max_tokens=max_tokens,\n            stop=None,  # <END> is cut client-side: Qwen writes it inside its reasoning, and a server-side stop would end the reply with empty content\n'
  '            extra_body=__import__("json").loads(os.environ.get("CYBENCH_EXTRA_BODY", "{}")) or None,\n        )'),
 ('model_response = response.choices[0].message.content or ""',
  'model_response = (response.choices[0].message.content or "").split("<END>")[0]'),
])
# local tokenizer fallback (model name is not a known OpenAI model)
p = root / "agent/models/non_helm.py"; s = p.read_text()
s = s.replace("tiktoken.encoding_for_model(model_name)", 'tiktoken.get_encoding("cl100k_base")'); p.write_text(s)

patch("agent/dataclasses/agent_spec.py", [
 ('    YI_LARGE = "01-ai/yi-large"\n', '    YI_LARGE = "01-ai/yi-large"\n    LOCAL = "openai/local"\n'),
])
p = root / "agent/dataclasses/agent_spec.py"; s = p.read_text()
if "LOCAL_REGISTERED" not in s:
    s += ('\n# LOCAL_REGISTERED\nimport os as _os\n'
          'TokenizerMapping.mapping[DeploymentName.LOCAL] = "openai/cl100k_base"\n'
          'ModelMapping.mapping[DeploymentName.LOCAL] = "openai/local"\n'
          'NonHELMMapping.mapping[DeploymentName.LOCAL] = _os.environ.get("CYBENCH_LOCAL_MODEL", "local")\n')
    p.write_text(s)

# run_task.sh: no TTY, no forced pull, let the container reach the host's model server
patch("run_task.sh", [
 ("\t-it \\\n\t--pull always \\\n", "\t--add-host host.docker.internal:host-gateway \\\n"),
])
PY
  run docker pull "$CYBENCH_IMAGE"
  say "Cybench ready. Subset file: $BASE/cybench_subset.txt"
  printf '%s\n' "${SUBSET[@]}" > "$BASE/cybench_subset.txt"
}

write_env(){   # $1 model, $2 engine, $3 address the server listens on
  local key extra hostip; key=$(api_key)
  # Not host.docker.internal: Cybench runs its own dockerd inside the agent container, whose docker0 (172.17.0.1) shadows host-gateway.
  if [ "$3" = 0.0.0.0 ]; then
    hostip=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p'); hostip=${hostip:-$(hostname -I | awk '{print $1}')}
  else hostip=$3; fi
  case "$1" in   # each model's recommended sampling
    qwen27b*) extra='{"top_k":20,"top_p":0.95}' ;;
    cybertiel*) extra='{"top_k":40,"min_p":0.05,"top_p":0.95}' ;;
    *) die "no Cybench sampling settings for '$1': add them to write_env" ;;
  esac
  ( umask 077; cat > "$CYBENCH_DIR/.env" <<EOF
OPENAI_API_KEY=$key
OPENAI_BASE_URL=http://$hostip:$PORT/v1
CYBENCH_LOCAL_MODEL=$(served "$1" "$2")
CYBENCH_TEMPERATURE=0.6
CYBENCH_EXTRA_BODY=$extra
EOF
  )
}

cybench_run(){
  local n="${1:-10}"
  [[ "$n" =~ ^[0-9]+$ ]] || die "--cybench-run takes a number of tasks, not '$n'"
  [ -d "$CYBENCH_DIR/.git" ] || die "run --cybench-setup first"
  read_active
  local m=$ACTIVE_M e=$ACTIVE_E
  # The agent container reaches the model over the host's LAN address, so a loopback-only server is out of its reach.
  case "$ACTIVE_B" in 127.*) die "the model listens on $ACTIVE_B only, which Cybench's agent container cannot reach: start it with BIND=0.0.0.0" ;; esac
  if [ "$DRY" = 1 ]; then
    say "(dry run: would write $CYBENCH_DIR/.env, move $CYBENCH_DIR/logs aside and run the first $n tasks on $m ($e))"; return; fi
  kcurl -fs --max-time 10 -o /dev/null "http://$(probe_host "$ACTIVE_B"):$PORT/v1/models" || die "model server is not answering"
  write_env "$m" "$e" "$ACTIVE_B"
  local out; out="$BASE/results/$m-$e-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$out"
  say "Cybench: $m, first $n tasks, unguided, $CYBENCH_ITER iterations -> $out"
  cd "$CYBENCH_DIR"
  # Cybench appends every run to ./logs; move earlier runs aside so this run's summary only sees its own results.
  if [ -n "$(ls -A logs 2>/dev/null)" ]; then mv logs "logs.prev-$(date +%Y%m%d-%H%M%S)"; fi; mkdir -p logs
  local i=0 t
  for t in "${SUBSET[@]}"; do
    i=$((i+1)); [ $i -le "$n" ] || break
    if [ -n "$CYBENCH_ONLY" ] && [[ ",$CYBENCH_ONLY," != *",$i,"* ]]; then continue; fi
    say "[$i/$n] $t"
    local start=$SECONDS rc=0 before; before=$(docker ps -aq --no-trunc | sort)
    timeout "$CYBENCH_TASK_TIMEOUT" ./run_task.sh --task_dir "$t" --max_iterations "$CYBENCH_ITER" \
      --unguided_mode --max_input_tokens "$CYBENCH_MAX_IN" --max_output_tokens "$CYBENCH_MAX_OUT" \
      --model openai/local > "$out/task-$i.log" 2>&1 || rc=$?
    printf '%s\t%s\t%s\t%s\n' "$i" "$t" "$rc" "$((SECONDS-start))" >> "$out/tasks.tsv"
    echo "    exit=$rc  $((SECONDS-start))s"
    docker rm -f cybench >/dev/null 2>&1 || true
    # A timeout kills run_task.sh before its own cleanup, leaving challenge containers that hold ports (e.g. 1337) for later tasks.
    # Spare the model server (labeled spark-llm=1).
    local c; for c in $(comm -13 <(echo "$before") <(docker ps -aq --no-trunc | sort)); do
      case "$(docker inspect -f '{{index .Config.Labels "spark-llm"}}' "$c" 2>/dev/null)" in 1) ;; *) docker rm -f "$c" >/dev/null 2>&1 || true;; esac
    done
  done
  mkdir -p "$out/logs"; cp -r "$CYBENCH_DIR/logs/." "$out/logs/" 2>/dev/null || true
  say "Done. Raw logs: $out   (summarize with: $0 --cybench-summary $out)"
}

cybench_summary(){
  python3 - "${1:?results dir}" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); tsv = d / "tasks.tsv"
rows = [l.rstrip("\n").split("\t") for l in tsv.read_text().splitlines()] if tsv.exists() else []
logs = {}
for lg in sorted((d / "logs").rglob("*.json"), key=lambda q: q.stat().st_mtime):
    try: j = json.loads(lg.read_text())
    except Exception: continue
    logs[(j.get("task") or {}).get("path") or lg.parent.name] = j      # newest wins
def find(task):
    for k, j in logs.items():
        if k == task or task.endswith(k) or k.endswith(task) or k.split("/")[-1] == task.split("/")[-1]: return j
solved = ran = 0
print(f"{'#':>2}  {'result':8} {'iters':>5} {'out_tok':>8} {'wall':>6}  task")
for i, task, rc, secs in rows:
    j = find(task); name = task.split("benchmark/")[-1]
    if j is None and rc == "124":   # wall-clock cap hit while the agent was still working: counts as unsolved
        ran += 1
        print(f"{i:>2}  {'TIMEOUT':8} {'-':>5} {'-':>8} {secs+'s':>6}  {name}  (hit the wall-clock cap before the iteration budget; counted as FAIL)"); continue
    if j is None:
        print(f"{i:>2}  {'NOT RUN':8} {'-':>5} {'-':>8} {secs+'s':>6}  {name}  (exit={rc}, no result log: setup/infra failure)"); continue
    sc = j.get("subtask_completions") or {}
    ok = bool(j.get("num_correct_subtasks")) or any((v or {}).get("score", 0) == 1 or (v or {}).get("is_correct") for v in sc.values())
    its = [it for v in sc.values() for it in (v or {}).get("iterations", [])]
    toks = (j.get("usage") or {}).get("output_tokens_used", 0)
    ran += 1; solved += ok
    print(f"{i:>2}  {'PASS' if ok else 'FAIL':8} {len(its):>5} {toks:>8} {secs+'s':>6}  {name}")
print(f"\nSolved {solved}/{ran} tasks that ran (timeouts count as failures); {len(rows)-ran} of {len(rows)} could not run (excluded).")
PY
}

# ---------------------------------------------------------------- cleanup
cleanup(){
  local level="${1:-}" n c i
  if [ -z "$level" ]; then
    echo "  1) stop      stop model servers"
    echo "  2) containers  stop + remove Docker images and Cybench containers"
    echo "  3) weights   containers + delete downloaded model weights"
    echo "  4) all       everything under $BASE (weights, PolyStrata, Cybench, results, venv, API key)"
    read -r -p "Level [1-4]: " n; case "$n" in 1) level=stop;; 2) level=containers;; 3) level=weights;; 4) level=all;; *) return;; esac
  fi
  case "$level" in stop|containers|weights|all) ;; *) die "unknown cleanup level '$level' (stop, containers, weights or all)" ;; esac
  if [ "$level" = all ]; then
    case "${BASE%/}" in ""|"${HOME%/}") die "refusing to delete '$BASE': point SPARK_LLM_HOME at a folder of its own" ;; esac
  fi
  say "Current usage:"; du -sh "$BASE" "$BASE"/* 2>/dev/null | sed 's/^/  /' || true
  confirm "Proceed with cleanup level '$level'? This cannot be undone." || { echo "aborted"; return; }
  stop_all
  [ "$level" = stop ] && return
  if command -v docker >/dev/null 2>&1; then
    for c in $(docker ps -a --format '{{.Names}}' | grep -iE 'cybench|shared_net' || true); do run -q docker rm -f "$c" || true; done
    run -qq docker network rm shared_net || true
    for i in "$VLLM_IMAGE" "$LLAMA_IMAGE" "$CYBENCH_IMAGE"; do run -qq docker rmi "$i" || true; done
  fi
  [ "$level" = containers ] && return
  run rm -rf "$BASE/models" "$PS_DATA"
  [ "$level" = weights ] && return
  run rm -rf "$BASE"
  [ "$DRY" = 1 ] || say "Removed $BASE"
}

# The menu runs every action as a separate run of this script: inside `a && b` (or any condition) bash ignores `set -e`
# for the whole function body, so a failed download would go on to "Installed" and a restart. A child process keeps
# its own `set -e`, and a failure returns to the menu instead of ending it.
self(){
  local pass=(); if [ "$YES" = 1 ]; then pass+=(--yes); fi; if [ "$DRY" = 1 ]; then pass+=(--dry-run); fi
  "$BASH" "$SELF" "${pass[@]}" "$@"
}
menu_do(){ self "$@" && return 0; local rc=$?; echo "  ($* failed with exit status $rc; see the messages above)"; return "$rc"; }

menu(){
  while true; do
    echo; echo " DGX Spark model manager      [active: $(cat "$ACTIVE" 2>/dev/null || echo none)]"
    local i=0 r k t e c; for r in "${MODELS[@]}"; do i=$((i+1)); IFS='|' read -r k e t _ <<<"$r"; printf '  %2d) %-48s %s\n' "$i" "$t" "$e"; done
    cat <<EOF
   s) Stop running model       t) Status          p) Speed test (tok/s)
   c) Set up Cybench           r) Run Cybench on the running model (10-task subset)
   x) Clean up                 q) Quit
EOF
    read -r -p "> " c || exit 0
    case "$c" in
      [0-9]*) [ "$c" -ge 1 ] 2>/dev/null && [ "$c" -le "${#MODELS[@]}" ] || { echo "?"; continue; }
              IFS='|' read -r k e _ <<<"${MODELS[$((c-1))]}"
              # install only what is missing (a start needs no free disk space), then start
              if [ -n "$(missing_for "$k" "$e")" ] && ! menu_do --install "$k" --engine "$e"; then continue; fi
              menu_do --start "$k" --engine "$e" || true ;;
      s) menu_do --stop || true ;; t) menu_do --status || true ;; p) menu_do --speed || true ;;
      c) menu_do --cybench-setup || true ;; r) menu_do --cybench-run 10 || true ;; x) menu_do --cleanup || true ;;
      q|Q) exit 0;; *) echo "?";;
    esac
  done
}

# ---------------------------------------------------------------- args
ACTION=""; ARG1=""; ENGINE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1;; --dry-run) DRY=1;;
    --engine) [ $# -gt 1 ] || die "--engine needs a value"; ENGINE="$2"; shift;;
    --install|--start|--cybench-run|--cleanup|--cybench-summary|--speed) ACTION="$1"; if [ $# -gt 1 ] && [[ "$2" != --* ]]; then ARG1="$2"; shift; fi;;
    --stop|--status|--cybench-setup|--list|--parity) ACTION="$1";;
    -h|--help) sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d'; exit 0;;
    *) die "unknown option $1";;
  esac; shift
done
case "$ACTION" in --install|--start) ;; *) [ -z "$ENGINE" ] || die "--engine goes with --install or --start only" ;; esac
case "$ACTION" in
  "") menu;;
  --list) list_models;;
  --install) install_model "$ARG1" "$ENGINE";;
  --start) start_model "$ARG1" "$ENGINE";;
  --stop) stop_all;; --status) status;;
  --speed) speed speed "$ARG1";; --parity) speed parity;;
  --cybench-setup) cybench_setup;; --cybench-run) cybench_run "${ARG1:-10}";;
  --cybench-summary) cybench_summary "$ARG1";;
  --cleanup) cleanup "$ARG1";;
esac
