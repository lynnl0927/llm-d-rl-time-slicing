package backends

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"time"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
)

const (
	// tpuRestoreTimeout bounds one restore invocation of the tpu-checkpoint
	// CLI: up to 600s waiting for the previous occupant's vfio groups to be
	// released, up to 600s for the slice-wide restore rendezvous, plus slack.
	// This is a backstop only — the CLI enforces its own, tighter timeouts
	// and exits on its own.
	tpuRestoreTimeout = 1300 * time.Second

	// tpuSnapshotTimeout bounds one checkpoint invocation (the CLI's
	// per-process timeout is 120s with one retry).
	tpuSnapshotTimeout = 300 * time.Second

	// tpuVfioDir must exist for this node to host TPU jobs; libtpu attaches
	// chips through their vfio iommu groups.
	tpuVfioDir = "/dev/vfio"
)

// TpuCheckpoint implements the Backend interface for TPU processes using the
// tpu-checkpoint CLI (libtpu control-pipe protocol), mirroring how the CUDA
// backend shells out to NVIDIA's cuda-checkpoint utility.
//
// Contract inherited from libtpu (see tools/tpu-checkpoint/README.md):
// restore is a slice-wide rendezvous — all of a job's TPU processes are
// restored concurrently in ONE CLI invocation, and a failed restore must
// NEVER be retried (the job faults instead; re-sending a pending RESTORE
// wedges libtpu's state machine).
type TpuCheckpoint struct {
	mu          sync.Mutex
	execCommand func(ctx context.Context, name string, args ...string) ([]byte, error)
	lookPath    func(string) (string, error)
	statPath    func(string) (os.FileInfo, error)
}

// NewTpuCheckpoint creates a new TpuCheckpoint backend.
func NewTpuCheckpoint() *TpuCheckpoint {
	return &TpuCheckpoint{
		execCommand: func(ctx context.Context, name string, args ...string) ([]byte, error) {
			return exec.CommandContext(ctx, name, args...).CombinedOutput()
		},
		lookPath: exec.LookPath,
		statPath: os.Stat,
	}
}

// Snapshot parks the TPU state of every process of the job in one CLI
// invocation.
func (t *TpuCheckpoint) Snapshot(ctx context.Context, req Request) error {
	pids := ExtractTpuPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for TPU snapshot")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	slog.InfoContext(ctx, "Snapshotting TPU PIDs", "pids", pids)
	t0 := time.Now()
	cmdCtx, cancel := context.WithTimeout(ctx, tpuSnapshotTimeout)
	defer cancel()
	if err := t.run(cmdCtx, "checkpoint", pids); err != nil {
		return fmt.Errorf("tpu-checkpoint checkpoint failed: %w", err)
	}
	slog.InfoContext(ctx, "tpu-checkpoint checkpoint took", "duration", time.Since(t0), "pids", pids)
	return nil
}

// Restore restores the TPU state of every process of the job in one CLI
// invocation. Exactly one attempt: on failure the error propagates and the
// state machine marks the job FAULTED — do not add retries at any layer.
func (t *TpuCheckpoint) Restore(ctx context.Context, req Request) error {
	pids := ExtractTpuPIDStrings(req.Config)
	if len(pids) == 0 {
		return fmt.Errorf("at least one PID is required for TPU restore")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	slog.InfoContext(ctx, "Restoring TPU PIDs", "pids", pids)
	t0 := time.Now()
	cmdCtx, cancel := context.WithTimeout(ctx, tpuRestoreTimeout)
	defer cancel()
	if err := t.run(cmdCtx, "restore", pids); err != nil {
		return fmt.Errorf("tpu-checkpoint restore failed (job must be treated as faulted, never re-issue a restore): %w", err)
	}
	slog.InfoContext(ctx, "tpu-checkpoint restore took", "duration", time.Since(t0), "pids", pids)
	return nil
}

// HealthCheck verifies the CLI is installed and the node exposes vfio groups.
func (t *TpuCheckpoint) HealthCheck(ctx context.Context) error {
	binaryPath := t.getTpuCheckpointPath()
	if _, err := t.lookPath(binaryPath); err != nil {
		return fmt.Errorf("tpu-checkpoint executable not found: %w", err)
	}
	if _, err := t.statPath(tpuVfioDir); err != nil {
		return fmt.Errorf("no %s on this node (not a TPU host?): %w", tpuVfioDir, err)
	}
	return nil
}

func (t *TpuCheckpoint) run(ctx context.Context, action string, pids []string) error {
	args := []string{"--action", action}
	for _, pid := range pids {
		args = append(args, "--pid", pid)
	}
	if out, err := t.execCommand(ctx, t.getTpuCheckpointPath(), args...); err != nil {
		return fmt.Errorf("command failed: %w, output: %s", err, string(out))
	}
	return nil
}

func (t *TpuCheckpoint) getTpuCheckpointPath() string {
	// First check if it's in the PATH
	if path, err := t.lookPath("tpu-checkpoint"); err == nil {
		return path
	}
	// Fallback to the location the agent image installs it to
	return "/usr/local/bin/tpu-checkpoint"
}

// ExtractTpuPIDStrings extracts PID strings from a TPU BackendConfig.
func ExtractTpuPIDStrings(config *pb.BackendConfig) []string {
	if config == nil {
		return nil
	}
	tpu := config.GetTpu()
	if tpu == nil {
		return nil
	}
	target := tpu.GetExplicitTarget()
	if target == nil {
		return nil
	}
	pids := make([]string, 0, len(target.GetPids()))
	for _, pid := range target.GetPids() {
		pids = append(pids, strconv.Itoa(int(pid)))
	}
	return pids
}

// BuildTpuConfig wraps PID strings into a TPU BackendConfig.
func BuildTpuConfig(pidStrings []string) *pb.BackendConfig {
	pids := make([]int32, 0, len(pidStrings))
	for _, s := range pidStrings {
		if pid, err := strconv.ParseInt(s, 10, 32); err == nil {
			pids = append(pids, int32(pid))
		}
	}
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Tpu{
			Tpu: &pb.TpuBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}
