package utils_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	snapshotutils "github.com/llm-d-incubation/llm-d-rl-time-slicing/pkg/snapshot-agent/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// writeTpuProc creates a fixture process under root: a libtpu control thread
// (if libtpuThread), a cgroup file bound to podUID, and optionally a dangling
// symlink fd pointing at a /dev/vfio group.
func writeTpuProc(t *testing.T, root string, pid int, libtpuThread bool, podUID string, vfio bool) {
	t.Helper()
	base := filepath.Join(root, fmt.Sprint(pid))
	taskDir := filepath.Join(base, "task", fmt.Sprint(pid+1))
	if err := os.MkdirAll(taskDir, 0o755); err != nil {
		t.Fatal(err)
	}
	comm := "python3\n"
	if libtpuThread {
		comm = "libtpu00030004\n"
	}
	if err := os.WriteFile(filepath.Join(taskDir, "comm"), []byte(comm), 0o644); err != nil {
		t.Fatal(err)
	}
	cgroup := fmt.Sprintf("0::/kubepods/burstable/pod%s/cont\n", podUID)
	if err := os.WriteFile(filepath.Join(base, "cgroup"), []byte(cgroup), 0o644); err != nil {
		t.Fatal(err)
	}
	fdDir := filepath.Join(base, "fd")
	if err := os.MkdirAll(fdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/dev/null", filepath.Join(fdDir, "0")); err != nil {
		t.Fatal(err)
	}
	if vfio {
		if err := os.Symlink("/dev/vfio/0", filepath.Join(fdDir, "7")); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGetPodTpuPIDs(t *testing.T) {
	root := t.TempDir()
	restoreProc := snapshotutils.SetProcRootForTest(root)
	defer restoreProc()

	const podUID = "tpu-pod-uid"
	writeTpuProc(t, root, 100, true, podUID, true)        // running TPU proc of our pod
	writeTpuProc(t, root, 200, true, podUID, false)       // checkpointed: thread but no vfio fd
	writeTpuProc(t, root, 300, true, "other-uid", true)   // other pod's TPU proc
	writeTpuProc(t, root, 400, false, podUID, false)      // plain process of our pod

	origGetK8sClient := snapshotutils.GetK8sClient
	defer func() { snapshotutils.GetK8sClient = origGetK8sClient }()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-ns",
			UID:       types.UID(podUID),
		},
	}
	snapshotutils.GetK8sClient = func() (kubernetes.Interface, error) {
		return fake.NewSimpleClientset(pod), nil
	}

	pids, err := snapshotutils.GetPodTpuPIDs(context.Background(), "test-pod", "test-ns")
	if err != nil {
		t.Fatalf("GetPodTpuPIDs() error = %v", err)
	}
	if want := []int{100}; !reflect.DeepEqual(pids, want) {
		t.Errorf("GetPodTpuPIDs() = %v, want %v", pids, want)
	}
}

func TestHasTpuProcesses(t *testing.T) {
	t.Run("RunningProc", func(t *testing.T) {
		root := t.TempDir()
		defer snapshotutils.SetProcRootForTest(root)()
		writeTpuProc(t, root, 100, true, "uid", true)

		got, err := snapshotutils.HasTpuProcesses(context.Background())
		if err != nil || !got {
			t.Errorf("HasTpuProcesses() = %v, %v; want true, nil", got, err)
		}
	})

	t.Run("OnlyCheckpointedProc", func(t *testing.T) {
		root := t.TempDir()
		defer snapshotutils.SetProcRootForTest(root)()
		writeTpuProc(t, root, 100, true, "uid", false)

		got, err := snapshotutils.HasTpuProcesses(context.Background())
		if err != nil || got {
			t.Errorf("HasTpuProcesses() = %v, %v; want false, nil (checkpointed proc holds no vfio fd)", got, err)
		}
	})

	t.Run("NoTpuProcs", func(t *testing.T) {
		root := t.TempDir()
		defer snapshotutils.SetProcRootForTest(root)()
		writeTpuProc(t, root, 100, false, "uid", false)

		got, err := snapshotutils.HasTpuProcesses(context.Background())
		if err != nil || got {
			t.Errorf("HasTpuProcesses() = %v, %v; want false, nil", got, err)
		}
	})
}
