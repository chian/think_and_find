#!/usr/bin/env bash
# Run inside tools/one_workflow.sh; all dependencies and caches stay under runs/.
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
task_destination=$(realpath -m -- "${1:?Supply an installation directory beneath runs/}")
case "$task_destination" in "$task_repo/runs/"*) ;; *) echo 'Lean dependencies belong beneath this checkout runs/.' >&2; exit 64 ;; esac
if [[ -z ${THINK_AND_FIND_WORKFLOW_RESOURCES-} ]]; then
    echo 'Use this repository tools/one_workflow.sh for the installation workload.' >&2
    exit 64
fi
task_version=v4.35.0
task_archive=lean-4.35.0-linux.tar.zst
task_digest=7289597a44543246c9227b8cae501767e095ce17f6d448b95fc4a8dc97268f6d
mkdir -p -- "$task_destination/dependencies" "$task_destination/cache" "$task_destination/elan"
export ELAN_HOME="$task_destination/elan"
export MATHLIB_CACHE_DIR="$task_destination/cache/mathlib"
task_lean="$task_destination/dependencies/lean-4.35.0-linux"
if [[ ! -x $task_lean/bin/lean ]]; then
    curl -fL "https://github.com/leanprover/lean4/releases/download/$task_version/$task_archive" \
        -o "$task_destination/dependencies/$task_archive"
    printf '%s  %s\n' "$task_digest" "$task_destination/dependencies/$task_archive" | sha256sum --check
    tar --zstd -xf "$task_destination/dependencies/$task_archive" -C "$task_destination/dependencies"
fi
export PATH="$task_lean/bin:$PATH"
"$task_lean/bin/lean" --version > "$task_destination/lean-version.txt"
task_imports='["Std"]'
task_paths='[]'
if [[ ${2-} == --with-mathlib ]]; then
    task_mathlib="$task_destination/dependencies/mathlib4"
    if [[ ! -d $task_mathlib/.git ]]; then
        git clone --depth 1 --branch "$task_version" https://github.com/leanprover-community/mathlib4.git "$task_mathlib"
    fi
    (
        cd -- "$task_mathlib"
        lake exe cache get Mathlib.Data.Real.Basic Mathlib.Tactic.Ring Mathlib.Tactic.FieldSimp \
            Mathlib.Tactic.NormNum Mathlib.Tactic.Linarith
        lake env printenv LEAN_PATH > "$task_destination/lean-path.txt"
        git rev-parse HEAD > "$task_destination/mathlib-commit.txt"
    )
    task_imports='["Std","Mathlib.Data.Real.Basic","Mathlib.Tactic.Ring","Mathlib.Tactic.FieldSimp","Mathlib.Tactic.NormNum","Mathlib.Tactic.Linarith"]'
    task_paths=$(jq -Rsc --arg prefix "$task_mathlib" \
        'split("\n")[0] | split(":") | map(if startswith("/") then . else $prefix + "/" + . end)' \
        "$task_destination/lean-path.txt")
fi
task_config="$task_destination/lean.json"
jq -n --arg executable "$task_lean/bin/lean" --arg version "$task_version" \
    --arg origin "https://github.com/leanprover/lean4/releases/tag/$task_version" \
    --argjson imports "$task_imports" --argjson lean_path "$task_paths" \
    '{executable:$executable,version:$version,toolchain_origin:$origin,imports:$imports,lean_path:$lean_path}' > "$task_config"
mkdir -p -- "$task_repo/runs/scientific_tools"
cp -- "$task_config" "$task_repo/runs/scientific_tools/lean.json"
printf 'Lean scientific-tool configuration: %s\n' "$task_config"
