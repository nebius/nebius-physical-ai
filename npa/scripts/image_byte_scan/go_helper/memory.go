// Derive a Go soft memory limit from the scanner's visible cgroup-v2 allowance.
package main

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"syscall"
)

// cgroupPath rejects traversal rather than normalizing an unexpected proc path.
func cgroupPath(value string) string {
	value = strings.NewReplacer(`\040`, " ", `\011`, "\t", `\012`, "\n", `\134`, `\`).Replace(value)
	if !filepath.IsAbs(value) || filepath.Clean(value) != value || strings.ContainsRune(value, 0) {
		return ""
	}
	return value
}

// unifiedCgroup returns the process membership, not the mount's root cgroup.
func unifiedCgroup(data []byte) string {
	for _, line := range strings.Split(string(data), "\n") {
		parts := strings.SplitN(line, ":", 3)
		if len(parts) == 3 && parts[0] == "0" && parts[1] == "" {
			return cgroupPath(parts[2])
		}
	}
	return ""
}

// cgroupMount maps membership through a cgroup-v2 mount, including subtree mounts.
func cgroupMount(line, group string) (string, string, string) {
	parts := strings.SplitN(line, " - ", 2)
	if len(parts) != 2 {
		return "", "", ""
	}
	fields, filesystem := strings.Fields(parts[0]), strings.Fields(parts[1])
	if len(fields) < 6 || len(filesystem) < 1 || filesystem[0] != "cgroup2" {
		return "", "", ""
	}
	root, mount := cgroupPath(fields[3]), cgroupPath(fields[4])
	if root == "" || mount == "" {
		return "", "", ""
	}
	relative, err := filepath.Rel(root, group)
	if err != nil || relative == ".." || strings.HasPrefix(relative, "../") {
		return "", "", ""
	}
	return filepath.Join(mount, relative), mount, root
}

// cgroupDirectory prefers the mount exposing the most ancestors of this process.
func cgroupDirectory(membership, mounts []byte) (string, string) {
	group := unifiedCgroup(membership)
	if group == "" {
		return "", ""
	}
	leaf, boundary, selectedRoot := "", "", ""
	for _, line := range strings.Split(string(mounts), "\n") {
		candidate, mount, root := cgroupMount(line, group)
		if candidate != "" && (selectedRoot == "" || len(root) < len(selectedRoot)) {
			leaf, boundary, selectedRoot = candidate, mount, root
		}
	}
	return leaf, boundary
}

// cgroupBytes accepts the kernel's unsigned decimal format without signed aliases.
func cgroupBytes(data []byte) (uint64, bool) {
	text := strings.TrimSpace(string(data))
	if text == "" || strings.IndexFunc(text, func(char rune) bool { return char < '0' || char > '9' }) >= 0 {
		return 0, false
	}
	value, err := strconv.ParseUint(text, 10, 64)
	return value, err == nil
}

// cgroupAllowance reserves ten percent of currently unused cgroup memory.
// Startup usage includes this helper's small initial heap; subtracting it again
// is conservative and avoids estimating how much of memory.current Go owns.
func cgroupAllowance(directory string, readFile func(string) ([]byte, error)) (int64, string) {
	maximum, err := readFile(filepath.Join(directory, "memory.max"))
	if err != nil || strings.TrimSpace(string(maximum)) == "max" {
		return 0, ""
	}
	limit, valid := cgroupBytes(maximum)
	if !valid {
		return 0, "cgroup_memory_limit_invalid"
	}
	if limit > math.MaxInt64 {
		return 0, ""
	}
	usage, err := readFile(filepath.Join(directory, "memory.current"))
	if os.IsNotExist(err) || errors.Is(err, syscall.ENODEV) {
		return 0, ""
	}
	current, valid := cgroupBytes(usage)
	if err != nil || !valid {
		return 0, "cgroup_memory_usage_unavailable"
	}
	if current >= limit {
		return 0, ""
	}
	remaining := limit - current
	return int64(remaining - remaining/10), ""
}

// cgroupMemoryLimit includes every readable visible ancestor, not just the leaf.
// Missing cgroup-v2 metadata preserves the existing runtime default. Once a
// finite limit is known, malformed usage cannot silently select a looser budget.
// Exhausted or too-small samples are advisory: current includes reclaimable cache.
func cgroupMemoryLimit(readFile func(string) ([]byte, error), owned uint64) (int64, string) {
	membership, memberErr := readFile("/proc/self/cgroup")
	mounts, mountErr := readFile("/proc/self/mountinfo")
	if memberErr != nil || mountErr != nil {
		return 0, ""
	}
	directory, boundary := cgroupDirectory(membership, mounts)
	var selected int64
	for directory != "" {
		limit, code := cgroupAllowance(directory, readFile)
		if code != "" {
			return 0, code
		}
		if uint64(limit) > owned && (selected == 0 || limit < selected) {
			selected = limit
		}
		if directory == boundary {
			break
		}
		directory = filepath.Dir(directory)
	}
	return selected, ""
}

// applyCgroupMemoryLimit never raises an explicitly smaller runtime limit.
func applyCgroupMemoryLimit(readFile func(string) ([]byte, error), owned uint64, setLimit func(int64) int64) string {
	limit, code := cgroupMemoryLimit(readFile, owned)
	if code != "" || limit == 0 {
		return code
	}
	if limit < setLimit(-1) {
		setLimit(limit)
	}
	return ""
}

// configureCgroupMemory runs before configuration or detector allocations.
func configureCgroupMemory() string {
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	return applyCgroupMemoryLimit(os.ReadFile, memory.Sys-memory.HeapReleased, debug.SetMemoryLimit)
}
