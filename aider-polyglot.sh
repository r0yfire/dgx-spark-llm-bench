#!/usr/bin/env bash
# aider-polyglot.sh - Aider's Polyglot coding benchmark (225 Exercism exercises in C++, Go, Java, JavaScript, Python and
# Rust) against the model spark-llm.sh is serving. The model edits the exercise's files from its instructions, the tests
# run, and on a failure it sees the test output and gets a second try: an agentic edit-test loop, scored as
# pass_rate_1 / pass_rate_2 on the same scale as Aider's public leaderboard.
#
#   aider-polyglot.sh --setup                 clone Aider and the exercises at their pinned commits, build the benchmark image
#   aider-polyglot.sh --run [N]               run on the active model: all 225 exercises, or N of them (Aider picks them at random)
#   aider-polyglot.sh --toolchains            run every language's tests once without a model (the image works on this CPU)
#   aider-polyglot.sh --stats [DIR]           Aider's report of a run (default: the latest one)
#
# A run continues where it stopped: --run picks up the latest unfinished run of the active model and engine, and Aider
# skips every exercise that already has a result. AIDER_NEW=1 starts a fresh one instead.
#
# Settings (environment):
#   SPARK_LLM_HOME    where spark-llm.sh keeps its state (default ~/spark-llm): .active, .active.json, .api_key, results/
#   PORT              the model server's port (default 8000)
#   AIDER_SUBSET      a file listing the exercises to run, one path per line (# comments allowed), e.g.
#                     aider-subset-60.txt beside this script: the fixed 60-exercise comparison set, 10 per language
#   AIDER_KEYWORDS    only exercises whose path contains one of these (comma-separated), e.g. python/exercises/practice/bowling
#   AIDER_LANGUAGES   only these languages (comma-separated), e.g. python,rust
#   AIDER_TRIES       attempts per exercise (default 2, as on the leaderboard)
#   AIDER_THREADS     exercises in parallel (default 1: the engines here serve one stream at a time)
#   AIDER_EDIT_FORMAT edit format (default diff: see below)
#   AIDER_NEW=1       start a new run even if an unfinished one exists
#   AIDER_THINKING    on (default, as for Cybench) or off (Inkling 2 Mini: its lowest effort, none). On, Qwen3.8-Flash-Next
#                     at its default effort writes ~25K tokens per first reply
#                     on these exercises (about 10 minutes each at ~80 tok/s, so ~38 hours for all 225)
#   AIDER_EFFORT      reasoning effort, passed as chat_template_kwargs.reasoning_effort (default: unset, the template's own
#                     default). Qwen3.8: low, medium or xhigh (default xhigh). Inkling 2 Mini: none, minimal, low, medium,
#                     high, xhigh, max or 0.0-0.99 (default 0.9, "high"). Needs thinking on.
#   AIDER_MAX_TOKENS  output limit per reply (default 32768)
#
# Model settings: the model is reached as openai/<served name> through OPENAI_API_BASE (Aider's documented route for an
# OpenAI-compatible server). Sampling is each model's recommended one, the same as spark-llm.sh's Cybench runs, with
# thinking on; max_tokens 32768 and a 1-hour request timeout leave room for long reasoning on a slow engine.
# Edit format "diff": both Qwen3 entries on Aider's leaderboard ran with it, Aider's built-in settings give it to every
# large Qwen model, and most leaderboard entries use it; "whole" would re-emit whole files and cost far more decode time.
#
# Safety: the benchmark runs model-written code, so it runs inside Aider's container (as upstream requires), with the
# 12 GB memory cap of Aider's docker.sh. The API key reaches the container as an environment variable copied from this
# process (docker run -e OPENAI_API_KEY, no value on the command line) and is never printed.
set -euo pipefail

# Pinned: Aider's main (the benchmark harness lives there; 2026-05-22) and the exercise set (2024-12-22).
AIDER_REPO="https://github.com/Aider-AI/aider.git"
AIDER_REF="5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_REPO="https://github.com/Aider-AI/polyglot-benchmark.git"
POLYGLOT_REF="7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"

BASE="${SPARK_LLM_HOME:-$HOME/spark-llm}"
PORT="${PORT:-8000}"
WORK="$BASE/aider-bench"                     # the Aider checkout (mounted at /aider) and the exercises
AIDER_DIR="$WORK/aider"
EXERCISES="$WORK/polyglot-benchmark"
IMAGE="spark-llm-aider-benchmark:${AIDER_REF:0:12}"
KEYFILE="$BASE/.api_key"; ACTIVE="$BASE/.active"; ACTIVE_META="$BASE/.active.json"
TRIES="${AIDER_TRIES:-2}"; THREADS="${AIDER_THREADS:-1}"; EDIT_FORMAT="${AIDER_EDIT_FORMAT:-diff}"
THINKING="${AIDER_THINKING:-on}"; MAX_TOKENS="${AIDER_MAX_TOKENS:-32768}"; EFFORT="${AIDER_EFFORT:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say(){ printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

checkout(){   # $1 repo URL, $2 commit, $3 dir: a clone at exactly that commit
  if [ ! -d "$3/.git" ]; then git clone -q "$1" "$3"; fi
  if [ "$(git -C "$3" rev-parse HEAD)" != "$2" ]; then
    git -C "$3" fetch -q origin "$2" 2>/dev/null || git -C "$3" fetch -q origin
    git -C "$3" -c advice.detachedHead=false checkout -q --force "$2"
  fi
  [ "$(git -C "$3" rev-parse HEAD)" = "$2" ] || die "$3 is not at $2"
  [ -z "$(git -C "$3" status --porcelain)" ] || die "$3 has local changes; the pinned code must run as is"
}

setup(){
  command -v docker >/dev/null || die "docker is not installed"
  mkdir -p "$WORK"
  say "Aider at $AIDER_REF"; checkout "$AIDER_REPO" "$AIDER_REF" "$AIDER_DIR"
  say "Polyglot exercises at $POLYGLOT_REF"; checkout "$POLYGLOT_REPO" "$POLYGLOT_REF" "$EXERCISES"
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then say "Image $IMAGE already built"; return; fi
  # benchmark/docker_build.sh's own command, under a tag that names the pinned commit
  say "Building $IMAGE from Aider's benchmark/Dockerfile (10-20 minutes)"
  mkdir -p "$BASE/logs"
  ( cd "$AIDER_DIR" && docker build --file benchmark/Dockerfile -t "$IMAGE" . ) > "$BASE/logs/aider-image-build.log" 2>&1 ||
    { tail -25 "$BASE/logs/aider-image-build.log"; die "image build failed: see $BASE/logs/aider-image-build.log"; }
  say "Image built ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19))"
}

meta(){   # $1 key -> its value in .active.json (empty when absent)
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$ACTIVE_META" "$1"
}

model_settings(){   # $1 model, $2 served name, $3 file: Aider model settings (YAML) for openai/<served>
  local temp=0.6 top_p=0.95 think x body=() kw=()
  case "$THINKING" in on) think=true ;; off) think=false ;; *) die "AIDER_THINKING must be on or off, not '$THINKING'" ;; esac
  [[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "AIDER_MAX_TOKENS must be a number of tokens"
  [ -z "$EFFORT" ] || [ "$THINKING" = on ] || die "AIDER_EFFORT sets the effort of thinking: it needs AIDER_THINKING=on"
  # Each model's recommended sampling, the same as spark-llm.sh's Cybench runs (write_env), and its template's thinking switches
  case "$1" in
    qwen27b*|flashnext*)   # Qwen3.8's template: enable_thinking, and reasoning_effort low | medium | xhigh (default xhigh)
      [ -z "$EFFORT" ] || [[ "$EFFORT" =~ ^(low|medium|xhigh)$ ]] ||
        die "Qwen3.8's chat template takes reasoning_effort low, medium or xhigh (default xhigh), not '$EFFORT'"
      body=("top_k: 20"); kw=("enable_thinking: $think")
      [ -z "$EFFORT" ] || kw+=("reasoning_effort: \"$EFFORT\"") ;;
    inkling*)   # Inkling 2 Mini's template has no enable_thinking: reasoning_effort none | minimal | low | medium | high |
                # xhigh | max, or 0.0-0.99 (default 0.9 = high); "none" is its lowest setting (it still reasons a little).
                # Sampling: its generation_config.json (temperature 1.0); its card and vLLM recipe give no other advice.
      temp=1.0; top_p=""
      if [ "$THINKING" = off ]; then kw=('reasoning_effort: "none"')
      elif [[ "$EFFORT" =~ ^(none|minimal|low|medium|high|xhigh|max)$ ]]; then kw=("reasoning_effort: \"$EFFORT\"")
      elif [[ "$EFFORT" =~ ^0(\.[0-9]{1,2})?$ ]]; then kw=("reasoning_effort: $EFFORT")
      elif [ -n "$EFFORT" ]; then die "Inkling 2 Mini's chat template takes reasoning_effort none, minimal, low, medium, high, xhigh, max or 0.0-0.99, not '$EFFORT'"
      fi ;;
    cybertiel*)
      [ -z "$EFFORT" ] || die "CyberTiel's chat template has no reasoning effort setting: unset AIDER_EFFORT"
      body=("top_k: 40" "min_p: 0.05") ;;
    *) die "no Aider sampling settings for '$1': add them to model_settings in $0" ;;
  esac
  {
    printf -- '- name: openai/%s\n  edit_format: %s\n  use_repo_map: true\n  use_temperature: %s\n  reasoning_tag: think\n' \
      "$2" "$EDIT_FORMAT" "$temp"
    printf '  extra_params:\n    max_tokens: %s\n    timeout: 3600\n' "$MAX_TOKENS"
    [ -z "$top_p" ] || printf '    top_p: %s\n' "$top_p"
    if [ ${#body[@]} -gt 0 ] || [ ${#kw[@]} -gt 0 ]; then
      printf '    extra_body:\n'
      for x in "${body[@]}"; do printf '      %s\n' "$x"; done
      if [ ${#kw[@]} -gt 0 ]; then
        printf '      chat_template_kwargs:\n'
        for x in "${kw[@]}"; do printf '        %s\n' "$x"; done
      fi
    fi
  } > "$3"
}

SUBSET_FILE=""; SUBSET_SHA=""; SUBSET=()
load_subset(){   # $1 file (as given, or beside this script): SUBSET = its exercise paths, each checked to name exactly one exercise
  local f=$1
  [ -f "$f" ] || f="$SCRIPT_DIR/$1"
  [ -f "$f" ] || die "AIDER_SUBSET: no file '$1' (here or beside $0)"
  mapfile -t SUBSET < <(sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//' "$f" | grep -v '^$')
  [ ${#SUBSET[@]} -gt 0 ] || die "AIDER_SUBSET: $f lists no exercises"
  # Aider selects by substring (--keywords), so every line must be a real exercise path that no other path contains
  python3 - "$EXERCISES" "${SUBSET[@]}" <<'PY' || die "AIDER_SUBSET: $f does not match the pinned exercises (see above)"
import glob, os, sys
root, wanted = sys.argv[1], sys.argv[2:]
paths = [os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "*", "exercises", "practice", "*")) if os.path.isdir(p)]
bad = [(w, [p for p in paths if w in p]) for w in wanted]
bad = [(w, m) for w, m in bad if m != [w]]
dups = sorted({w for w in wanted if wanted.count(w) > 1})
for w, m in bad:
    print(f"  {w}: matches {m or 'no exercise'}")
for w in dups:
    print(f"  {w}: listed twice")
sys.exit(1 if bad or dups else 0)
PY
  SUBSET_FILE=$(basename "$f")
  SUBSET_SHA=$(printf '%s\n' "${SUBSET[@]}" | sha256sum | cut -c1-64)
}

HOST_IP=""
host_ip(){   # sets HOST_IP: the address the container uses to reach the model server on this host
  local b; b=$(meta bind); b=${b:-0.0.0.0}
  case "$b" in 127.*) die "the model listens on $b only, which the benchmark container cannot reach: start it with BIND=0.0.0.0" ;; esac
  if [ "$b" = 0.0.0.0 ]; then
    b=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p'); b=${b:-$(hostname -I | awk '{print $1}')}
  fi
  [ -n "$b" ] || die "cannot tell this host's LAN address"
  HOST_IP=$b
}

SETTINGS=""   # JSON of everything that defines a run (set by run): a run is only continued with the same settings
run_dir(){   # $1 model, $2 engine, $3 served name -> the latest unfinished run with the same served name and settings, or a new one
  local d i
  if [ "${AIDER_NEW:-0}" != 1 ]; then
    local runs=("$BASE/results/aider-$1-$2-"*)   # timestamped names: glob order is age order
    for ((i = ${#runs[@]} - 1; i >= 0; i--)); do
      d=${runs[i]}
      [ -d "$d" ] && [ ! -e "$d/.done" ] && [ -s "$d/run.json" ] || continue
      python3 - "$d/run.json" "$3" "$SETTINGS" <<'PY' && { echo "$d"; return; }
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get("served") == sys.argv[2] and r.get("settings") == json.loads(sys.argv[3]) else 1)
PY
    done
  fi
  echo "$BASE/results/aider-$1-$2-$(date +%Y%m%d-%H%M%S)"
}

bench(){   # $1 results dir, $2.. benchmark.py arguments (after the run dir): run inside the image, as root, results chowned back
  local out=$1; shift
  local key; key=$(tr -d '[:space:]' < "$KEYFILE")
  OPENAI_API_KEY="$key" docker run --rm --memory=12g --memory-swap=12g \
    -v "$AIDER_DIR:/aider" -v "$EXERCISES:/benchmarks/polyglot-benchmark:ro" -v "$out:/out" \
    -e OPENAI_API_KEY -e "OPENAI_API_BASE=${API_BASE:-http://127.0.0.1:$PORT/v1}" \
    -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/benchmarks -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
    "$IMAGE" bash -c 'rc=0; ./benchmark/benchmark.py "$@" || rc=$?; chown -R "$HOST_UID:$HOST_GID" /out; exit $rc' bench /out/polyglot "$@"
}

count_results(){ if [ -d "$1/polyglot" ]; then find "$1/polyglot" -name .aider.results.json | wc -l; else echo 0; fi; }

run(){   # $1 number of exercises (empty: all)
  local n=${1:-} m e served url d sel args
  [ -s "$ACTIVE" ] && [ -s "$ACTIVE_META" ] || die "no model is running: start one with spark-llm.sh --start"
  docker image inspect "$IMAGE" >/dev/null 2>&1 && [ -d "$EXERCISES/.git" ] || die "run $0 --setup first"
  [ -z "$n" ] || [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "--run takes a number of exercises, not '$n'"
  m=$(meta model); e=$(meta engine); served=$(meta served)
  [ -n "$m" ] && [ -n "$e" ] && [ -n "$served" ] || die "$ACTIVE_META does not name the model, engine and served name"
  host_ip; url="http://$HOST_IP:$PORT/v1"
  local key; key=$(tr -d '[:space:]' < "$KEYFILE")
  printf 'header = "Authorization: Bearer %s"\n' "$key" | curl -fs -K - --max-time 10 -o /dev/null "$url/models" ||
    die "the model server does not answer at $url"
  # which exercises: a subset file, or keywords / languages / a random N (a subset excludes the others)
  local expected=225 tmp
  sel=()
  if [ -n "${AIDER_SUBSET:-}" ]; then
    [ -z "${AIDER_KEYWORDS:-}${AIDER_LANGUAGES:-}$n" ] || die "AIDER_SUBSET already chooses the exercises: drop AIDER_KEYWORDS, AIDER_LANGUAGES and N"
    load_subset "$AIDER_SUBSET"; expected=${#SUBSET[@]}
    sel+=(--keywords "$(IFS=,; echo "${SUBSET[*]}")")
  fi
  [ -z "${AIDER_KEYWORDS:-}" ] || sel+=(--keywords "$AIDER_KEYWORDS")
  [ -z "${AIDER_LANGUAGES:-}" ] || sel+=(--languages "$AIDER_LANGUAGES")
  [ -z "$n" ] || sel+=(--num-tests "$n")
  tmp=$(mktemp); model_settings "$m" "$served" "$tmp"   # validates thinking / effort / max tokens before any directory is made
  SETTINGS=$(python3 - "$THINKING" "$EFFORT" "$MAX_TOKENS" "$EDIT_FORMAT" "$TRIES" "$SUBSET_FILE" "$SUBSET_SHA" \
                       "${AIDER_KEYWORDS:-}" "${AIDER_LANGUAGES:-}" "$n" <<'PY'
import json, sys
think, effort, maxtok, fmt, tries, sub, subsha, kw, langs, n = sys.argv[1:]
print(json.dumps({"thinking": think, "reasoning_effort": effort or None, "max_tokens": int(maxtok), "edit_format": fmt,
                  "tries": int(tries), "subset": sub or None, "subset_sha256": subsha or None, "keywords": kw or None,
                  "languages": langs or None, "num_tests": int(n) if n else None}, sort_keys=True))
PY
)
  d=$(run_dir "$m" "$e" "$served"); mkdir -p "$d"; mv "$tmp" "$d/model-settings.yml"
  # what this run is, recorded once; a later --run continues it only with the same served name and settings (run_dir)
  if [ ! -s "$d/run.json" ]; then
    python3 - "$d/run.json" "$ACTIVE_META" "$m" "$e" "$served" "$AIDER_REF" "$POLYGLOT_REF" "$SETTINGS" <<'PY'
import json, sys, time
f, meta, m, e, served, aref, pref, settings = sys.argv[1:]
json.dump({"model": m, "engine": e, "served": served, "aider_ref": aref, "polyglot_ref": pref,
           "settings": json.loads(settings), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "server": json.load(open(meta))},
          open(f, "w"), indent=1)
PY
  fi
  args=(--model "openai/$served" --edit-format "$EDIT_FORMAT" --threads "$THREADS" --tries "$TRIES"
        --read-model-settings /out/model-settings.yml --exercises-dir polyglot-benchmark "${sel[@]}")
  say "Aider Polyglot: $m ($e) as openai/$served, edit format $EDIT_FORMAT, thinking $THINKING${EFFORT:+ (effort $EFFORT)}," \
      "${SUBSET_FILE:+subset $SUBSET_FILE ($expected exercises), }$TRIES tries, $THREADS thread(s)" \
      "-> $d ($(count_results "$d") exercises already done)"
  local t0 rc st; t0=$(date +%s)
  # progress: Aider prints its running totals after every exercise
  set +e
  API_BASE="$url" bench "$d" "${args[@]}" 2>&1 | tee -a "$d/benchmark.log" | grep -a -E '^(  pass_rate|  percent_cases|  completed_tests|fnames:|Copying)'
  st=("${PIPESTATUS[@]}"); set -e
  rc=${st[0]}
  echo "$(( $(date +%s) - t0 ))" >> "$d/elapsed_s"
  report "$d"
  [ "$rc" = 0 ] || die "benchmark.py exited with $rc (see $d/benchmark.log); run --run again to continue"
  if [ -z "$n" ] && [ -z "${AIDER_KEYWORDS:-}${AIDER_LANGUAGES:-}" ] && [ "$(count_results "$d")" -ge "$expected" ]; then touch "$d/.done"; fi
}

report(){   # $1 results dir: Aider's own stats, then the total wall time over every session of the run
  local d=$1 total
  [ -d "$d/polyglot" ] || die "$d holds no run"
  docker run --rm -v "$AIDER_DIR:/aider" -v "$d:/out:ro" -e AIDER_DOCKER=1 -e AIDER_BENCHMARK_DIR=/out "$IMAGE" \
    ./benchmark/benchmark.py --stats /out/polyglot > "$d/stats.txt" 2>&1 || true
  grep -a -E '^\s*(test_cases|completed_tests|model|edit_format|pass_rate_[0-9]|percent_cases_well_formed|error_outputs|num_malformed_responses|user_asks|exhausted_context_windows|test_timeouts|seconds_per_case|prompt_tokens|completion_tokens):' "$d/stats.txt" || tail -20 "$d/stats.txt"
  total=$(awk '{s += $1} END {print s + 0}' "$d/elapsed_s" 2>/dev/null || echo 0)
  printf '  total_wall_time: %dh %02dm (%d s over all sessions)\n' $((total / 3600)) $((total % 3600 / 60)) "$total" | tee "$d/total_time.txt"
}

toolchains(){   # every language's tests on one unsolved exercise each, no model: shows the image works on this CPU
  docker image inspect "$IMAGE" >/dev/null 2>&1 || die "run $0 --setup first"
  local d; d="$BASE/results/aider-toolchains-$(date +%Y%m%d-%H%M%S)"; mkdir -p "$d"
  say "Toolchain check (no model; the stub solutions are expected to FAIL their tests) -> $d"
  bench "$d" --model none --no-aider --tries 1 --threads 1 --exercises-dir polyglot-benchmark \
    --keywords "cpp/exercises/practice/clock,go/exercises/practice/wordy,java/exercises/practice/house,javascript/exercises/practice/triangle,python/exercises/practice/phone-number,rust/exercises/practice/acronym" \
    > "$d/benchmark.log" 2>&1 || true
  local f n=0
  for f in $(find "$d/polyglot" -name .aider.results.json 2>/dev/null | sort); do
    n=$((n + 1))
    python3 - "$f" "$(dirname "$f")/.aider.chat.history.md" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1]))
try:
    blocks = re.findall(r"```\n(.*?)\n```", open(sys.argv[2], errors="replace").read(), re.S)
except OSError:
    blocks = []
tail = [l for l in (blocks[-1] if blocks else "").splitlines() if l.strip()][-2:]
print(f"  {r.get('testcase', '?'):<14} tests ran, outcome {r.get('tests_outcomes')} | " + " / ".join(t.strip()[:90] for t in tail))
PY
  done
  echo "  $n of 6 languages produced a test result (a missing toolchain crashes the run instead: see $d/benchmark.log)"
  grep -a -E "command not found|exec format error|FileNotFoundError|Traceback \(most recent" "$d/benchmark.log" | head -5 || true
}

case "${1:-}" in
  --setup) setup ;;
  --run) run "${2:-}" ;;
  --toolchains) toolchains ;;
  --stats) d=${2:-}
           if [ -z "$d" ]; then   # the most recently started run (any model), not a toolchain check
             for r in "$BASE/results/aider-"*/run.json; do [ -e "$r" ] && [ "$r" -nt "${d:-/nonexistent}/run.json" ] && d=$(dirname "$r"); done
           fi
           [ -n "$d" ] || die "no runs yet"; report "$d" ;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
