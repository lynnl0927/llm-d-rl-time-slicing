package utils

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// TPU process discovery. There is no NVML equivalent for TPUs; a process is
// "on the accelerator" when it (a) runs libtpu with checkpointing enabled —
// visible as a control thread named "libtpu{RRRRSSSS}" in /proc/<pid>/task —
// and (b) holds an open /dev/vfio/<group> fd (libtpu attaches chips through
// their vfio iommu groups). A checkpointed process keeps its control thread
// but releases its vfio fds, and a process that never initialized the TPU has
// neither, so requiring both yields exactly the RUNNING set.
//
// Discovery runs on the host (the agent DaemonSet is privileged with
// hostPID), so PIDs are host-namespace PIDs — the same namespace the
// tpu-checkpoint CLI targets.

var tpuControlThreadRe = regexp.MustCompile(`^libtpu[0-9a-fA-F]{8}$`)

// procRoot is a package var so tests can point discovery at a fixture tree.
var procRoot = "/proc"

// GetPodTpuPIDs returns the host PIDs of all TPU-attached processes belonging
// to the specified pod. Drop-in replacement for GetPodPIDs (assigned over it
// when the agent runs with ACCELERATOR_TYPE=tpu).
func GetPodTpuPIDs(ctx context.Context, podName, namespace string) ([]int, error) {
	podUID, err := getPodUID(ctx, podName, namespace)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod UID: %w", err)
	}

	candidates, err := listTpuProcesses()
	if err != nil {
		return nil, err
	}

	var pids []int
	for _, pid := range candidates {
		inCgroup, err := IsPIDInPodCgroupInternal(fmt.Sprintf("%s/%d/cgroup", procRoot, pid), podUID)
		if err != nil || !inCgroup {
			continue
		}
		if holdsVfioFd(pid) {
			pids = append(pids, pid)
		}
	}
	return pids, nil
}

// HasTpuProcesses reports whether any process on the node is attached to the
// TPU. Drop-in replacement for HasGPUProcesses on TPU nodes.
func HasTpuProcesses(ctx context.Context) (bool, error) {
	candidates, err := listTpuProcesses()
	if err != nil {
		return false, err
	}
	for _, pid := range candidates {
		if holdsVfioFd(pid) {
			return true, nil
		}
	}
	return false, nil
}

// listTpuProcesses returns every PID with a libtpu control thread.
func listTpuProcesses() ([]int, error) {
	procs, err := os.ReadDir(procRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s: %w", procRoot, err)
	}

	var pids []int
	for _, proc := range procs {
		pid, err := strconv.Atoi(proc.Name())
		if err != nil {
			continue
		}
		tasks, err := os.ReadDir(filepath.Join(procRoot, proc.Name(), "task"))
		if err != nil {
			continue // process gone or not ours to read
		}
		for _, task := range tasks {
			comm, err := os.ReadFile(filepath.Join(procRoot, proc.Name(), "task", task.Name(), "comm"))
			if err != nil {
				continue
			}
			if tpuControlThreadRe.MatchString(strings.TrimSpace(string(comm))) {
				pids = append(pids, pid)
				break
			}
		}
	}
	return pids, nil
}

// holdsVfioFd reports whether the process has an open /dev/vfio/<group> fd.
func holdsVfioFd(pid int) bool {
	fdDir := fmt.Sprintf("%s/%d/fd", procRoot, pid)
	fds, err := os.ReadDir(fdDir)
	if err != nil {
		return false
	}
	for _, fd := range fds {
		target, err := os.Readlink(filepath.Join(fdDir, fd.Name()))
		if err != nil {
			continue
		}
		if strings.HasPrefix(target, "/dev/vfio/") {
			return true
		}
	}
	return false
}
