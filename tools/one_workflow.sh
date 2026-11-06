#!/usr/bin/env bash
# One Think and Find workload per Codex window, with whole-workload CPU/RAM bounds.
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$task_repo"
if [[ $# -lt 3 || $2 != -- ]]; then
    echo 'Usage: bash tools/one_workflow.sh runs/<unique-run> -- command arguments...' >&2
    exit 64
fi
task_run=$(realpath -m -- "$1")
shift 2
case "$task_run" in "$task_repo/runs/"*) ;; *) echo 'Output must be under think_and_find/runs/.' >&2; exit 64 ;; esac
task_window=${CODEX_THREAD_ID-}
if [[ ! $task_window =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]]; then
    echo 'A stable CODEX_THREAD_ID is required; do not invent one per run.' >&2
    exit 64
fi
task_key=${task_window//-/}
task_service="think-and-find-workflow-$task_key.service"
task_slice="thinkandfindwindow$task_key.slice"
task_slot="$task_repo/runs/workflow-windows/$task_window"
mkdir -p -- "$task_slot"
exec 9>"$task_slot/workflow.lock"
if ! flock -n 9; then echo 'This Codex window already owns a Think and Find workload.' >&2; exit 75; fi

# A window can already have a workload launched from another checkout. Inspect
# its actual service and processes; a different application name is not a slot.
task_units=$(systemctl --user list-units --all --type=service --plain --no-legend --no-pager '*workflow*.service')
while read -r task_unit task_rest; do
    [[ -n $task_unit ]] || continue
    task_environment=$(systemctl --user show "$task_unit" --property=Environment --value)
    [[ " $task_environment " == *" CODEX_THREAD_ID=$task_window "* ]] || continue
    task_state=$(systemctl --user show "$task_unit" --property=ActiveState --value)
    case "$task_state" in
        active|activating|deactivating|reloading)
            echo "This Codex window still owns $task_unit." >&2; exit 75 ;;
    esac
    task_group=$(systemctl --user show "$task_unit" --property=ControlGroup --value)
    if [[ -n $task_group && -d /sys/fs/cgroup$task_group ]]; then
        task_members=$(find "/sys/fs/cgroup$task_group" -name cgroup.procs -exec cat {} +)
        if [[ -n $task_members ]]; then echo 'The prior workload still has processes.' >&2; exit 75; fi
    fi
done <<< "$task_units"
task_state=$(systemctl --user show "$task_service" --property=ActiveState --value)
case "$task_state" in active|activating|deactivating|reloading) echo 'The prior workload is still active.' >&2; exit 75 ;; esac
task_group=$(systemctl --user show "$task_slice" --property=ControlGroup --value)
if [[ -n $task_group && -d /sys/fs/cgroup$task_group ]]; then
    task_members=$(find "/sys/fs/cgroup$task_group" -name cgroup.procs -exec cat {} +)
    if [[ -n $task_members ]]; then echo 'The prior workload still has processes.' >&2; exit 75; fi
fi
systemctl --user reset-failed "$task_service" 2>/dev/null || true
mkdir -p -- "$(dirname -- "$task_run")"
if ! mkdir -- "$task_run"; then echo 'A fresh run directory is required.' >&2; exit 64; fi
task_python=$(command -v python3)
task_status=0
systemd-run --user --quiet --wait --pipe --collect --service-type=exec --expand-environment=no \
    --unit="$task_service" --slice="$task_slice" \
    --property=CPUQuota=100% --property=MemoryMax=5368709120 \
    --property=MemorySwapMax=0 --property=TasksMax=infinity \
    --property=KillMode=control-group --working-directory="$task_repo" \
    --setenv="PYTHONPATH=$task_repo" --setenv=PYTHONDONTWRITEBYTECODE=1 \
    --setenv="CODEX_THREAD_ID=$task_window" \
    "$task_python" -B "$task_repo/tools/workflow_resources.py" "$task_run" -- "$@" \
    >"$task_run/stdout.log" 2>"$task_run/stderr.log" || task_status=$?
task_group=$(systemctl --user show "$task_slice" --property=ControlGroup --value)
if [[ -n $task_group && -d /sys/fs/cgroup$task_group ]]; then
    task_members=$(find "/sys/fs/cgroup$task_group" -name cgroup.procs -exec cat {} +)
    if [[ -n $task_members ]]; then echo 'Workload processes remain; completion is unconfirmed.' >&2; exit 76; fi
fi
if [[ $task_status == 0 && ! -f "$task_run/workflow-resources.json" ]]; then
    echo 'Workload resource verification is missing.' >&2
    task_status=76
fi
cat -- "$task_run/stdout.log"
cat -- "$task_run/stderr.log" >&2
exit "$task_status"
