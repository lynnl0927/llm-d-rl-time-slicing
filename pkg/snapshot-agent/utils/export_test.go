package utils

// SetProcRootForTest points TPU process discovery at a fixture tree and
// returns a restore function.
func SetProcRootForTest(root string) func() {
	old := procRoot
	procRoot = root
	return func() { procRoot = old }
}
