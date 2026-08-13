package backends_test

import (
	"context"
	"fmt"
	"os"
	"slices"
	"testing"

	pb "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/api/v1alpha1"
	"github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/backends"
)

func tpuConfig(pids ...int32) *pb.BackendConfig {
	return &pb.BackendConfig{
		Backend: &pb.BackendConfig_Tpu{
			Tpu: &pb.TpuBackendConfig{
				ExplicitTarget: &pb.ProcessTarget{Pids: pids},
			},
		},
	}
}

func TestNewTpuCheckpoint(t *testing.T) {
	if backends.NewTpuCheckpoint() == nil {
		t.Fatal("NewTpuCheckpoint returned nil")
	}
}

func TestTpuSnapshot(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: tpuConfig(123, 456),
		},
		{
			name:        "ExecFailure",
			config:      tpuConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
		{
			name:        "NoPIDs",
			config:      tpuConfig(),
			expectedErr: true,
		},
		{
			name:        "NilConfig",
			config:      nil,
			expectedErr: true,
		},
		{
			name:        "CudaConfigRejected",
			config:      cudaConfig(123),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewTpuCheckpoint()
			calls := 0
			c.SetExecCommand(func(_ context.Context, _ string, _ ...string) ([]byte, error) {
				calls++
				return nil, tt.execErr
			})

			err := c.Snapshot(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Snapshot() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if tt.execErr != nil && calls != 1 {
				t.Errorf("Snapshot() invoked the CLI %d times, want exactly 1 (the CLI owns retries)", calls)
			}
		})
	}
}

// TestTpuSnapshotArgs pins the CLI contract: one invocation per job carrying
// every PID (restore is a slice-wide rendezvous, so the CLI must see the
// whole mesh at once).
func TestTpuSnapshotArgs(t *testing.T) {
	c := backends.NewTpuCheckpoint()
	var gotArgs []string
	c.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
		gotArgs = args
		return nil, nil
	})

	if err := c.Snapshot(context.Background(), backends.Request{JobID: "j", Config: tpuConfig(11, 22)}); err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}
	want := []string{"--action", "checkpoint", "--pid", "11", "--pid", "22"}
	if !slices.Equal(gotArgs, want) {
		t.Errorf("Snapshot() args = %v, want %v", gotArgs, want)
	}
}

func TestTpuRestore(t *testing.T) {
	tests := []struct {
		name        string
		config      *pb.BackendConfig
		execErr     error
		expectedErr bool
	}{
		{
			name:   "Success",
			config: tpuConfig(123),
		},
		{
			name:        "NoPIDs",
			config:      tpuConfig(),
			expectedErr: true,
		},
		{
			name:        "ExecFailure",
			config:      tpuConfig(123),
			execErr:     fmt.Errorf("exec error"),
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewTpuCheckpoint()
			calls := 0
			var gotArgs []string
			c.SetExecCommand(func(_ context.Context, _ string, args ...string) ([]byte, error) {
				calls++
				gotArgs = args
				return nil, tt.execErr
			})

			err := c.Restore(context.Background(), backends.Request{JobID: "test-job", Config: tt.config})
			if (err != nil) != tt.expectedErr {
				t.Errorf("Restore() error = %v, expectedErr %v", err, tt.expectedErr)
			}
			if len(gotArgs) > 0 && (gotArgs[0] != "--action" || gotArgs[1] != "restore") {
				t.Errorf("Restore() args = %v, want --action restore first", gotArgs)
			}
			// A failed restore must never be retried: a duplicate RESTORE
			// request wedges libtpu's state machine.
			if tt.execErr != nil && calls != 1 {
				t.Errorf("Restore() invoked the CLI %d times after failure, want exactly 1 (never retry)", calls)
			}
		})
	}
}

func TestTpuHealthCheck(t *testing.T) {
	tests := []struct {
		name        string
		lookPathErr error
		statErr     error
		expectedErr bool
	}{
		{
			name: "Success",
		},
		{
			name:        "MissingCLI",
			lookPathErr: fmt.Errorf("not found"),
			expectedErr: true,
		},
		{
			name:        "MissingVfio",
			statErr:     os.ErrNotExist,
			expectedErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := backends.NewTpuCheckpoint()
			c.SetLookPath(func(path string) (string, error) {
				return path, tt.lookPathErr
			})
			c.SetStatPath(func(string) (os.FileInfo, error) {
				return nil, tt.statErr
			})

			err := c.HealthCheck(context.Background())
			if (err != nil) != tt.expectedErr {
				t.Errorf("HealthCheck() error = %v, expectedErr %v", err, tt.expectedErr)
			}
		})
	}
}
