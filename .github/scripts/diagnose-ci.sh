#!/usr/bin/env bash
# Temporary Linux runner instrumentation. Usage: diagnose-ci.sh OUTPUT COMMAND [ARG...]
set -uo pipefail
output=$1
shift
mkdir -p "$output"
output=$(cd "$output" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FIRECROWN_DIAGNOSTICS_DIR="$output"
export PYTHONPATH="$script_dir/ci_diagnostics${PYTHONPATH:+:$PYTHONPATH}"
export PYTEST_PLUGINS="pytest_diagnostics${PYTEST_PLUGINS:+,$PYTEST_PLUGINS}"
export PYTHONFAULTHANDLER=1

# No environment dumps or process command lines: either can contain credentials.
kernel_snapshot() {
    local messages
    if messages=$(sudo -n dmesg --ctime 2>&1); then
        printf '%s\n' "$messages" | grep -Ei 'oom|out of memory|killed process|segfault|trap' || true
    else
        printf '[DEBUG-ci-resources] kernel log unavailable: %s\n' "$messages"
    fi
}
snapshot() {
    printf '\n[DEBUG-ci-resources] sample %s\n' "$(date -u +%FT%TZ)"
    cat /proc/loadavg /proc/meminfo
    for resource in cpu memory io; do
        cat "/proc/pressure/$resource" 2>/dev/null || true
    done
    # Sample the process cgroup and its ancestors: limits can live above it.
    cgroup_path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
    cgroup_dir="/sys/fs/cgroup$cgroup_path"
    while [[ "$cgroup_dir" == /sys/fs/cgroup* ]]; do
        for metric in memory.current memory.peak memory.max memory.events memory.swap.current pids.current pids.max pids.events; do
            if [[ -r "$cgroup_dir/$metric" ]]; then
                printf '%s: ' "$cgroup_dir/$metric"
                tr '\n' ' ' < "$cgroup_dir/$metric"
                printf '\n'
            fi
        done
        [[ "$cgroup_dir" == /sys/fs/cgroup ]] && break
        cgroup_dir=$(dirname "$cgroup_dir")
    done
    ps -eo pid,ppid,pgid,nlwp,rss,vsz,pcpu,stat,comm --sort=-rss
}
kernel_snapshot > "$output/kernel-before.log"
{
    uname -a
    printf 'CPUs: '; getconf _NPROCESSORS_ONLN
    ulimit -a
    cat /proc/self/cgroup
} > "$output/runner.log" 2>&1
(
    while true; do
        snapshot
        sleep 2
    done
) > "$output/resources.log" 2>&1 &
monitor_pid=$!
# Invoked by the EXIT trap below, including when the test command fails.
# shellcheck disable=SC2329
cleanup() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    snapshot >> "$output/resources.log" 2>&1
    kernel_snapshot > "$output/kernel-after.log"
}
trap 'cleanup' EXIT
printf '[DEBUG-ci-resources] command_start %s\n' "$(date -u +%FT%TZ)" > "$output/command.log"
"$@" 2>&1 | tee "$output/tests.log"
result=${PIPESTATUS[0]}
printf '[DEBUG-ci-resources] command_end %s exit=%s\n' "$(date -u +%FT%TZ)" "$result" >> "$output/command.log"
exit "$result"
