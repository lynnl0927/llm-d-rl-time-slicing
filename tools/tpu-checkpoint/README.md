# tpu-checkpoint

The TPU sibling of NVIDIA's `cuda-checkpoint`: a single-file, stdlib-only
Python CLI that checkpoints and restores the TPU state of **running**
processes through libtpu's v2 control-pipe protocol. The snapshot agent's
`tpu` backend shells out to it, exactly the way the `cuda` backend shells out
to the `cuda-checkpoint` binary.

## Requirements

- Targets must run with `LIBTPU_CHECKPOINTING_ENABLED=true` (each TPU process
  then spawns a `libtpu{RRRRSSSS}` control thread whose name encodes its
  request/response pipe FDs).
- The CLI must run on the **host** as root in the host PID namespace (the
  snapshot agent DaemonSet already is): it talks to targets via
  `/proc/<pid>/fd/<N>` and gates restores on host `/dev/vfio/*`.
- Python 3.11+ (stdlib only).

## Usage

```bash
# Park the TPU state of one job (all of its TPU processes in one invocation):
tpu-checkpoint --action checkpoint --pid 1234 --pid 1235 ... [--timeout 120]

# Bring it back (gates on /dev/vfio release first):
tpu-checkpoint --action restore --pid 1234 --pid 1235 ... \
    [--timeout 600] [--vfio-gate-timeout 600]

# Inspect:
tpu-checkpoint --action status --pid 1234
```

Logs stream to stderr; the last stdout line is a JSON summary
(`{"action": ..., "ok": true, ...}`). Exit code 0 on success.

## Non-negotiable contracts

- **One invocation per job.** Restore is a slice-wide rendezvous inside
  libtpu: every mesh member's RESTORE must be pending simultaneously. All of a
  job's PIDs go into a single `--action restore` call, which issues them
  concurrently.
- **Never retry a restore.** A timed-out-but-pending RESTORE is still waiting
  in the rendezvous; re-sending injects a duplicate request that wedges
  libtpu's state machine. The CLI hard-codes `retries=0` for restore, and no
  caller (Go backend, state machine, controller) may add retries on top.
- **A checkpointed process must issue no TPU ops** until restored. The
  workload driver guarantees this by quiescing (draining device ops /
  aborting in-flight requests) before yielding its time-slice lock.
- **Restore gates on vfio release.** The previous occupant's chips are freed
  host-side a few seconds after its checkpoint completes; restoring into a
  busy iommu group poisons the driver. The CLI polls every `/dev/vfio/<n>`
  until openable before issuing the restore (default 600 s budget, tunable
  via `--vfio-gate-timeout` or `TS_VFIO_GATE_TIMEOUT_S`).
