#!/usr/bin/env python3
# Copyright 2025 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tpu-checkpoint: checkpoint/restore TPU state of running processes via libtpu.

The TPU sibling of NVIDIA's cuda-checkpoint utility, invoked by the snapshot
agent's "tpu" backend. Speaks the libtpu v2 gVisor tpu_control pipe protocol:
with LIBTPU_CHECKPOINTING_ENABLED=true each TPU process spawns a control
thread named "libtpu{RRRRSSSS}" (hex request-write / response-read pipe FDs,
visible in /proc/<pid>/task/*/comm). We send 4-byte-BE length-prefixed
tpu_control.proto messages through /proc/<pid>/fd/<N>.

Designed to run on the HOST (privileged, host PID namespace), so all targets
are explicit host PIDs -- there is deliberately no host-wide discovery, which
would sweep processes of every time-sliced job on the node.

Contracts (violating these wedges or poisons libtpu -- do not "fix" them):
  * Checkpoint parks the process's TPU state (state=detached). A checkpointed
    process must issue NO TPU ops until restored, or it aborts with
    "enqueueProgram ... Current state: TearedDown".
  * Restore is a slice-wide rendezvous: every mesh member's RESTORE must be
    pending simultaneously (a lone request parks until its peers arrive).
    Therefore restore is always issued concurrently to all PIDs of a job in
    ONE invocation, with retries=0 and a long timeout. NEVER retry a restore:
    a timed-out-but-pending RESTORE is still waiting in the barrier, and
    re-sending injects a duplicate request that wedges libtpu's state machine.
  * Restore reattaches chips via their vfio iommu groups, which the host side
    releases a few seconds AFTER the previous occupant's checkpoint returns.
    Firing Restore into an EBUSY group fails AND poisons the driver
    ("Reattach failed during Restore"), so restore first gates on every
    /dev/vfio group on the host becoming openable (wait_vfio_free).

Usage:
  tpu-checkpoint --action checkpoint --pid P1 [--pid P2 ...] [--timeout 120]
  tpu-checkpoint --action restore    --pid P1 [--pid P2 ...] [--timeout 600]
                 [--vfio-gate-timeout 600]
  tpu-checkpoint --action status     --pid P1 [--pid P2 ...]

Logs go to stderr; the last stdout line is a JSON summary. Exit 0 on success.
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import select
import struct
import sys
import time

_THREAD_RE = re.compile(r"^libtpu([0-9a-fA-F]{4})([0-9a-fA-F]{4})$")
_ACTIONS = {"Checkpoint": 2, "Restore": 3}
_STATES = {0: "unspecified", 1: "running", 2: "locked", 3: "detached", 4: "restoring", 5: "faulted"}


def _log(msg):
    print(f"[tpu-checkpoint] {msg}", file=sys.stderr, flush=True)


def _varint(val):
    out = b""
    while val >= 0x80:
        out += bytes([val & 0x7F | 0x80])
        val >>= 7
    return out + bytes([val])


def _read_varint(data, i):
    val, shift = 0, 0
    while True:
        b = data[i]
        val |= (b & 0x7F) << shift
        i += 1
        if b < 0x80:
            return val, i
        shift += 7


def get_pipes_for_pids(pids):
    """Return [(pid, tid, req_write_fd, rsp_read_fd)] for the given host PIDs.

    Exactly one libtpu control thread is expected per TPU process. PIDs whose
    control thread is missing (LIBTPU_CHECKPOINTING_ENABLED off, no TPU init,
    or process gone) are reported in the second return value.
    """
    pipes, missing = [], []
    for pid in pids:
        found = False
        for comm in glob.glob(f"/proc/{pid}/task/[0-9]*/comm"):
            try:
                with open(comm) as f:
                    name = f.read().strip()
            except OSError:
                continue
            m = _THREAD_RE.match(name)
            if m:
                tid = int(comm.split("/")[4])
                pipes.append((pid, tid, int(m.group(1), 16), int(m.group(2), 16)))
                found = True
        if not found:
            missing.append(pid)
    pipes.sort()
    _log(f"found {len(pipes)} libtpu control pipes for {len(pids)} PIDs: {[(p, t) for p, t, _, _ in pipes]}")
    return pipes, missing


def get_vfio_fds(pid):
    """Return the /dev/vfio/<group> paths held open by a process."""
    held = []
    for fd in glob.glob(f"/proc/{pid}/fd/[0-9]*"):
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if re.fullmatch(r"/dev/vfio/\d+", target):
            held.append(target)
    return sorted(set(held))


def scan_vfio_holders():
    """Host-wide scan: which processes hold which /dev/vfio group fds.

    Returns {"/dev/vfio/N": ["pid(comm)", ...]}. Runs on the host (hostPID), so
    on a shared node this attributes holds across ALL time-sliced jobs — the
    in-band answer to "who is holding the vfio lock" when a gate waits or a
    checkpoint leaves groups busy. A full /proc fd sweep of a fat libtpu
    process costs O(100ms); callers rate-limit to every few seconds.
    """
    holders = {}
    for fddir in glob.glob("/proc/[0-9]*/fd"):
        pid = fddir.split("/")[2]
        held = set()
        for fd in glob.glob(f"{fddir}/[0-9]*"):
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if re.fullmatch(r"/dev/vfio/\d+", target):
                held.add(target)
        if held:
            try:
                with open(f"/proc/{pid}/comm") as f:
                    comm = f.read().strip()
            except OSError:
                comm = "?"
            for g in held:
                holders.setdefault(g, []).append(f"{pid}({comm})")
    return {g: sorted(v) for g, v in sorted(holders.items())}


def wait_vfio_free(timeout_s, poll_s=0.25):
    """Block until every /dev/vfio iommu group on this host is openable.

    The previous occupant's chip release lags its logical yield by seconds
    (host-side hold by the HAL server / tpu-plugin daemon). Gating here keeps
    Restore from firing into EBUSY groups, and measures the release latency.
    Raises after timeout rather than letting Restore poison the driver.
    """
    groups = sorted(glob.glob("/dev/vfio/[0-9]*"))
    if not groups:
        _log("vfio gate: no /dev/vfio groups visible on this host; skipping")
        return 0.0
    t0 = time.time()
    last_holder_log = 0.0
    while True:
        busy = []
        for g in groups:
            try:
                fd = os.open(g, os.O_RDWR)
                os.close(fd)
            except OSError:
                busy.append(g)
        if not busy:
            waited = time.time() - t0
            if waited > 0.5:
                _log(f"vfio gate: waited {waited:.2f}s for group release")
            return waited
        now = time.time()
        # Attribute the hold while we wait (first hit, then every ~5s): which
        # PID/comm of which job is pinning each busy group.
        if now - last_holder_log >= 5.0:
            holders = scan_vfio_holders()
            _log(f"vfio gate: waiting {now - t0:.1f}s; busy={ {g: holders.get(g, ['<no fd holder: kernel-side hold?>']) for g in busy} }")
            last_holder_log = now
        if now - t0 > timeout_s:
            holders = scan_vfio_holders()
            raise RuntimeError(
                f"vfio gate: groups still busy after {timeout_s}s: "
                f"{ {g: holders.get(g, ['<no fd holder: kernel-side hold?>']) for g in busy} } "
                "- refusing to Restore into busy chips"
            )
        time.sleep(poll_s)


def _control(pipe, method, timeout=120):
    """Send one tpu_control action and wait for the response."""
    pid, _tid, req_fd, rsp_fd = pipe
    body = _varint((1 << 3) | 0) + _varint(_ACTIONS[method])
    body += _varint((2 << 3) | 0) + _varint(int(timeout))
    req = os.open(f"/proc/{pid}/fd/{req_fd}", os.O_WRONLY)
    rsp = os.open(f"/proc/{pid}/fd/{rsp_fd}", os.O_RDONLY)
    try:
        os.write(req, struct.pack(">I", len(body)) + body)
        deadline = time.time() + timeout

        # poll() instead of select(): select() rejects fd numbers >= 1024,
        # easily exceeded in fat processes.
        poller = select.poll()
        poller.register(rsp, select.POLLIN)

        def read_exact(n):
            buf = b""
            while len(buf) < n:
                r = poller.poll(max(0.0, deadline - time.time()) * 1000)
                if not r:
                    raise TimeoutError(f"pid={pid} no {method} response within {timeout}s")
                chunk = os.read(rsp, n - len(buf))
                if not chunk:
                    raise OSError(f"pid={pid} response pipe closed")
                buf += chunk
            return buf

        (size,) = struct.unpack(">I", read_exact(4))
        data = read_exact(size)
    finally:
        os.close(req)
        os.close(rsp)

    success, state, err, i = False, 0, "", 0
    while i < len(data):
        tag, i = _read_varint(data, i)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _read_varint(data, i)
            if field == 1:
                success = bool(val)
            elif field == 2:
                state = val
        elif wire == 2:
            ln, i = _read_varint(data, i)
            if field == 3:
                err = data[i : i + ln].decode(errors="replace")
            i += ln
        else:
            raise ValueError(f"unsupported wire type {wire}")
    if not success:
        raise RuntimeError(f"{method} pid={pid} failed: state={_STATES.get(state, state)} {err}")
    return _STATES.get(state, state)


def _run_control(pipe, method, retries=0, backoff=2.0, timeout=120):
    # NOTE on Restore: callers always pass retries=0 (see module docstring).
    pid = pipe[0]
    last = None
    for attempt in range(retries + 1):
        t0 = time.time()
        _log(f"-> {method} pid={pid} (attempt {attempt + 1}/{retries + 1})...")
        try:
            state = _control(pipe, method, timeout)
            _log(f"<- {method} pid={pid} OK state={state} in {time.time() - t0:.2f}s")
            return
        except Exception as e:  # noqa: BLE001 - every failure mode is retried/reported identically
            last = str(e)
            _log(f"<- {method} pid={pid} FAILED in {time.time() - t0:.2f}s: {last}")
            if attempt < retries:
                time.sleep(backoff)
    raise RuntimeError(f"{method} pid={pid} failed after {retries + 1} attempt(s): {last}")


def _fan_out(pipes, method, retries, timeout):
    """Run Checkpoint/Restore concurrently on every target process.

    Concurrency is mandatory for Restore (rendezvous semantics) and harmless
    for Checkpoint. Raises if any process ultimately fails.
    """
    _log(f"initiating {method} on {len(pipes)} processes (retries={retries}, timeout={timeout}s)...")
    t0 = time.time()
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pipes)) as ex:
        futs = {ex.submit(_run_control, p, method, retries, timeout=timeout): p for p in pipes}
        for fut in concurrent.futures.as_completed(futs):
            pipe = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 - collected per-PID and re-raised below
                failures[pipe[0]] = str(e)
    ok = len(pipes) - len(failures)
    _log(f"{method}: {ok}/{len(pipes)} ok in {time.time() - t0:.2f}s")
    if failures:
        raise RuntimeError(f"{method} failed on {len(failures)}/{len(pipes)} processes: {failures}")


def clear_lockfiles(pids):
    """Remove each target container's /tmp/libtpu_lockfile (via /proc/<pid>/root).

    A stale lockfile blocks the next libtpu init in that container. Running on
    the host, the container's /tmp is only reachable through the process's
    root fs view.
    """
    seen = set()
    for pid in pids:
        path = f"/proc/{pid}/root/tmp/libtpu_lockfile"
        try:
            real = os.path.realpath(path)
        except OSError:
            real = path
        if real in seen:
            continue
        seen.add(real)
        try:
            if os.path.exists(path):
                os.remove(path)
                _log(f"removed stale {path}")
        except OSError as e:
            _log(f"could not remove {path}: {e}")


def do_checkpoint(pids, timeout):
    pipes, missing = get_pipes_for_pids(pids)
    if missing:
        # A partial checkpoint leaves some processes attached to the chips, so
        # the job set must be complete before we touch anything.
        raise RuntimeError(
            f"PIDs without a libtpu control thread: {missing} "
            "(LIBTPU_CHECKPOINTING_ENABLED not on, no TPU init, or process gone)"
        )
    _fan_out(pipes, "Checkpoint", retries=1, timeout=timeout)
    clear_lockfiles(pids)
    # Post-checkpoint the parked processes should release their vfio groups
    # within seconds; report who still holds what so release lag (and any
    # third-party holder) is visible in the agent log.
    holders = scan_vfio_holders()
    _log(f"vfio holders after checkpoint: {holders or 'none'}")
    return {"processes": len(pipes), "vfio_holders_after": holders}


def do_restore(pids, timeout, vfio_gate_timeout):
    pipes, missing = get_pipes_for_pids(pids)
    if missing:
        raise RuntimeError(f"PIDs without a libtpu control thread: {missing} - cannot restore a partial mesh")
    # Gate ONCE before any process enters Restore: early members' Restore
    # opens their own vfio groups while parked waiting for siblings, so
    # gating per-process would deadlock against the job's own mesh.
    gate_wait = wait_vfio_free(vfio_gate_timeout)
    _fan_out(pipes, "Restore", retries=0, timeout=timeout)
    holders = scan_vfio_holders()
    _log(f"vfio holders after restore: {holders or 'none'}")
    return {"processes": len(pipes), "vfio_gate_wait_s": round(gate_wait, 2), "vfio_holders_after": holders}


def do_status(pids):
    procs = {}
    for pid in pids:
        pipes, missing = get_pipes_for_pids([pid])
        procs[pid] = {
            "alive": os.path.exists(f"/proc/{pid}"),
            "has_libtpu_thread": pid not in missing and bool(pipes),
            "vfio_fds": get_vfio_fds(pid),
        }
    free_groups, busy_groups = [], []
    for g in sorted(glob.glob("/dev/vfio/[0-9]*")):
        try:
            fd = os.open(g, os.O_RDWR)
            os.close(fd)
            free_groups.append(g)
        except OSError:
            busy_groups.append(g)
    return {"processes": procs, "vfio_free": free_groups, "vfio_busy": busy_groups}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tpu-checkpoint", description=__doc__.split("\n\n")[0])
    parser.add_argument("--action", required=True, choices=["checkpoint", "restore", "status"])
    parser.add_argument(
        "--pid", dest="pids", type=int, action="append", required=True, help="target host PID (repeatable)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-process control timeout in seconds (default: 120 for checkpoint, 600 for restore)",
    )
    parser.add_argument(
        "--vfio-gate-timeout",
        type=float,
        default=float(os.environ.get("TS_VFIO_GATE_TIMEOUT_S", "600")),
        help="max seconds to wait for /dev/vfio groups to be released before restore (default 600)",
    )
    args = parser.parse_args(argv)
    pids = sorted(set(args.pids))

    t0 = time.time()
    result = {"action": args.action, "pids": pids, "ok": False}
    try:
        if args.action == "checkpoint":
            result.update(do_checkpoint(pids, args.timeout or 120))
        elif args.action == "restore":
            result.update(do_restore(pids, args.timeout or 600, args.vfio_gate_timeout))
        else:
            result.update(do_status(pids))
        result["ok"] = True
    except Exception as e:  # noqa: BLE001 - single reporting point for the agent
        result["error"] = str(e)
        _log(f"FAILED: {e}")
    result["elapsed_s"] = round(time.time() - t0, 2)
    print(json.dumps(result), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
