#!/usr/bin/env bash
# Supervisor for the 14-day dealiasing campaign.
#
# 30_campaign.py is resumable (JSONL ledger, fsync'd per cell), so the recovery
# strategy for any crash -- CUDA fault, driver reset, OOM-kill, power cut -- is
# simply to run it again: it skips completed cells and picks up where it left
# off. This loop does that, with backoff so a cell that fails instantly cannot
# spin.
#
#   ./run_campaign.sh            supervise until the queue is complete
#   crontab @reboot              restart automatically after a power cut
#
# Safe to invoke twice: the lock below makes the second invocation a no-op.

set -u

SRC="/home/hubi/research/useful/papers/paper7_certified_spectral_pinn/src"
RES="/home/hubi/research/useful/papers/paper7_certified_spectral_pinn/results"
PY="/home/hubi/spde/venv/bin/python3"
LOG="$RES/30_campaign_supervisor.log"
LOCK="$RES/30_campaign.lock"

mkdir -p "$RES"

# single instance: @reboot and a manual start must not both take the GPU
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "$(date '+%F %T') another campaign instance holds the lock; exiting" >>"$LOG"
    exit 0
fi

log() { echo "$(date '+%F %T') $*" >>"$LOG"; }

log "=== supervisor start (pid $$) ==="

fails=0
while true; do
    log "launching 30_campaign.py"
    t0=$SECONDS
    "$PY" "$SRC/30_campaign.py" >>"$RES/30_campaign.out" 2>&1
    rc=$?
    ran=$((SECONDS - t0))

    # a run that lasted long enough to finish cells was making progress, so an
    # occasional later crash must not count toward the give-up threshold
    [ $ran -gt 600 ] && fails=0

    if [ $rc -eq 0 ]; then
        log "campaign exited cleanly (queue complete)"
        break
    fi

    # 130/143 = Ctrl-C / SIGTERM: a deliberate stop, do not fight it
    if [ $rc -eq 130 ] || [ $rc -eq 143 ]; then
        log "terminated by signal (rc=$rc); not restarting"
        break
    fi

    fails=$((fails + 1))
    if [ $fails -ge 40 ]; then
        log "40 consecutive failures; giving up (see 30_campaign.out)"
        break
    fi
    # 1,2,4,...,capped at 15 min
    backoff=$((60 * (1 << (fails < 4 ? fails - 1 : 4))))
    [ $backoff -gt 900 ] && backoff=900
    log "campaign died rc=$rc (failure $fails); retrying in ${backoff}s"
    sleep $backoff
done

log "=== supervisor exit ==="
