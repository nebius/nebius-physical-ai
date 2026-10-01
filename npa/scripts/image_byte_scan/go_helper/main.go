// Direct, whole-fragment Gitleaks bridge. No archive paths or matching bytes are
// emitted. The archive scanner supplies exact bytes and handles path-only rules.
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"sync"
	"syscall"
	"unsafe"

	"github.com/rs/zerolog"
	"github.com/spf13/viper"
	"github.com/zricethezav/gitleaks/v8/config"
	"github.com/zricethezav/gitleaks/v8/detect"
)

const detectorVersion = "8.28.0"
const processContainment = "seccomp-process-group-v1"
const maxMemoryBytes uint64 = 12 * 1024 * 1024 * 1024
const detectorMemoryHeadroom uint64 = 4 * 1024 * 1024 * 1024

// Current measurements peak at 9.85x payload bytes for one record. Round that
// observation up and derive admission from the hard address-space ceiling. The
// ceiling remains the fail-closed backstop if a future detector expands more.
const detectorPayloadExpansion uint64 = 10
const maxRecordBytes = (maxMemoryBytes - detectorMemoryHeadroom) / detectorPayloadExpansion
const completeRecordLimit uint64 = 1 << 30
const maxRecordFindings = 4096
const maxDetectionWorkers = 64

var mappedRecordUnmap = syscall.Munmap
var mappedRecordClose = func(file *os.File) error { return file.Close() }
var mappedRecordStat = func(file *os.File) (os.FileInfo, error) { return file.Stat() }
var mappedRecordDetect = detectMappedRecord

type sockFilter struct {
	Code uint16
	Jt   uint8
	Jf   uint8
	K    uint32
}

type sockFilterProgram struct {
	Length uint16
	Filter *sockFilter
}

const (
	bpfLoadWordAbsolute = 0x20
	bpfJumpEqual        = 0x15
	bpfJumpSet          = 0x45
	bpfReturn           = 0x06

	seccompDataSyscall = 0
	seccompDataArch    = 4
	seccompDataArg0    = 16

	auditArchX8664         = 0xc000003e
	seccompReturnKill      = 0x80000000
	seccompReturnErrno     = 0x00050000
	seccompReturnAllow     = 0x7fff0000
	prGetSeccomp           = 21
	prSetNoNewPrivileges   = 38
	prGetNoNewPrivileges   = 39
	seccompSetModeFilter   = 1
	seccompFilterFlagSync  = 1
	syscallSeccomp         = 317
	syscallClone           = 56
	syscallSetProcessGroup = 109
	syscallSetSession      = 112
	syscallUnshare         = 272
	syscallSetNamespace    = 308
	syscallClone3          = 435
	x32SyscallBit          = 0x40000000
	cloneNamespaceFlags    = 0x7e020080
)

const (
	processContainmentProbeArgument          = "--containment-probe"
	inheritedProcessContainmentProbeArgument = "--inherited-containment-probe"
)

func processContainmentFilter() []sockFilter {
	filter := []sockFilter{
		{Code: bpfLoadWordAbsolute, K: seccompDataArch},
		{Code: bpfJumpEqual, Jt: 1, K: auditArchX8664},
		{Code: bpfReturn, K: seccompReturnKill},
		{Code: bpfLoadWordAbsolute, K: seccompDataSyscall},
	}
	forbidden := []uint32{
		syscallSetProcessGroup,
		syscallSetSession,
		syscallUnshare,
		syscallSetNamespace,
	}
	forbiddenJumps := make([]int, 0, len(forbidden))
	forbiddenJumps = append(forbiddenJumps, len(filter))
	filter = append(filter, sockFilter{Code: bpfJumpSet, K: x32SyscallBit})
	for _, number := range forbidden {
		forbiddenJumps = append(forbiddenJumps, len(filter))
		filter = append(filter, sockFilter{Code: bpfJumpEqual, K: number})
	}
	clone3Jump := len(filter)
	filter = append(filter, sockFilter{Code: bpfJumpEqual, K: syscallClone3})
	filter = append(
		filter,
		sockFilter{Code: bpfJumpEqual, Jf: 2, K: syscallClone},
		sockFilter{Code: bpfLoadWordAbsolute, K: seccompDataArg0},
		sockFilter{Code: bpfJumpSet, Jt: 1, K: cloneNamespaceFlags},
		sockFilter{Code: bpfReturn, K: seccompReturnAllow},
		sockFilter{Code: bpfReturn, K: seccompReturnErrno | uint32(syscall.EPERM)},
		sockFilter{Code: bpfReturn, K: seccompReturnErrno | uint32(syscall.ENOSYS)},
	)
	errnoIndex := len(filter) - 2
	for _, index := range forbiddenJumps {
		filter[index].Jt = uint8(errnoIndex - index - 1)
	}
	filter[clone3Jump].Jt = uint8(len(filter) - clone3Jump - 2)
	return filter
}

func installProcessContainment() error {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		return fmt.Errorf("unsupported containment platform")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if _, _, errno := syscall.RawSyscall6(
		syscall.SYS_PRCTL,
		prSetNoNewPrivileges,
		1,
		0,
		0,
		0,
		0,
	); errno != 0 {
		return fmt.Errorf("set no-new-privileges: %w", errno)
	}
	filter := processContainmentFilter()
	program := sockFilterProgram{
		Length: uint16(len(filter)),
		Filter: &filter[0],
	}
	result, _, errno := syscall.RawSyscall6(
		syscallSeccomp,
		seccompSetModeFilter,
		seccompFilterFlagSync,
		uintptr(unsafe.Pointer(&program)),
		0,
		0,
		0,
	)
	runtime.KeepAlive(filter)
	if errno != 0 {
		return fmt.Errorf("install process containment: %w", errno)
	}
	if result != 0 {
		return fmt.Errorf("synchronize process containment: thread %d", result)
	}
	return nil
}

type threadContainmentResult struct {
	noNewPrivileges   uintptr
	noNewPrivilegeErr syscall.Errno
	unshareErr        syscall.Errno
}

func inspectContainedThread() threadContainmentResult {
	noNewPrivileges, _, noNewPrivilegeErr := syscall.RawSyscall(
		syscall.SYS_PRCTL,
		prGetNoNewPrivileges,
		0,
		0,
	)
	_, _, unshareErr := syscall.RawSyscall(syscallUnshare, 0, 0, 0)
	return threadContainmentResult{
		noNewPrivileges:   noNewPrivileges,
		noNewPrivilegeErr: noNewPrivilegeErr,
		unshareErr:        unshareErr,
	}
}

func (result threadContainmentResult) valid() bool {
	return result.noNewPrivilegeErr == 0 &&
		result.noNewPrivileges == 1 &&
		result.unshareErr == syscall.EPERM
}

type seccompFilterCount struct {
	value int
	valid bool
}

func currentThreadSeccompFilterCount() seccompFilterCount {
	status, err := os.ReadFile("/proc/thread-self/status")
	if err != nil {
		return seccompFilterCount{}
	}
	return seccompFilterCountFromStatus(status)
}

func seccompFilterCountFromStatus(status []byte) seccompFilterCount {
	prefix := []byte("Seccomp_filters:")
	found := seccompFilterCount{}
	for _, line := range bytes.Split(status, []byte{'\n'}) {
		if !bytes.HasPrefix(line, prefix) {
			continue
		}
		if found.valid {
			return seccompFilterCount{}
		}
		value, conversionErr := strconv.Atoi(
			string(bytes.TrimSpace(line[len(prefix):])),
		)
		if conversionErr != nil || value < 0 {
			return seccompFilterCount{}
		}
		found = seccompFilterCount{value: value, valid: true}
	}
	return found
}

type processContainmentResult struct {
	mode              uintptr
	modeErr           syscall.Errno
	thread            threadContainmentResult
	sessionErr        syscall.Errno
	groupErr          syscall.Errno
	x32Err            syscall.Errno
	setNamespaceErr   syscall.Errno
	clone3Err         syscall.Errno
	cloneNamespaceErr syscall.Errno
}

func inspectProcessContainment() processContainmentResult {
	mode, _, modeErr := syscall.RawSyscall(
		syscall.SYS_PRCTL,
		prGetSeccomp,
		0,
		0,
	)
	thread := inspectContainedThread()
	_, _, sessionErr := syscall.RawSyscall(syscall.SYS_SETSID, 0, 0, 0)
	_, _, groupErr := syscall.RawSyscall(syscall.SYS_SETPGID, 0, 0, 0)
	_, _, x32Err := syscall.RawSyscall(
		syscallSetProcessGroup|x32SyscallBit,
		0,
		0,
		0,
	)
	_, _, setNamespaceErr := syscall.RawSyscall(
		syscallSetNamespace,
		^uintptr(0),
		0,
		0,
	)
	_, _, clone3Err := syscall.RawSyscall(syscallClone3, 0, 0, 0)
	_, _, cloneNamespaceErr := syscall.RawSyscall6(
		syscallClone,
		uintptr(0x20000000|0x00010000|uint32(syscall.SIGCHLD)),
		0,
		0,
		0,
		0,
		0,
	)
	return processContainmentResult{
		mode:              mode,
		modeErr:           modeErr,
		thread:            thread,
		sessionErr:        sessionErr,
		groupErr:          groupErr,
		x32Err:            x32Err,
		setNamespaceErr:   setNamespaceErr,
		clone3Err:         clone3Err,
		cloneNamespaceErr: cloneNamespaceErr,
	}
}

func (result processContainmentResult) valid() bool {
	return result.modeErr == 0 &&
		result.mode == 2 &&
		result.thread.valid() &&
		result.sessionErr == syscall.EPERM &&
		result.groupErr == syscall.EPERM &&
		result.x32Err == syscall.EPERM &&
		result.setNamespaceErr == syscall.EPERM &&
		result.clone3Err == syscall.ENOSYS &&
		result.cloneNamespaceErr == syscall.EPERM
}

type preexistingThreadContainmentResult struct {
	before      seccompFilterCount
	after       seccompFilterCount
	containment threadContainmentResult
}

func (result preexistingThreadContainmentResult) valid() bool {
	return result.before.valid &&
		result.after.valid &&
		result.after.value == result.before.value+1 &&
		result.containment.valid()
}

func startPreexistingThreadProbe() (
	chan struct{},
	chan struct{},
	chan preexistingThreadContainmentResult,
) {
	ready := make(chan struct{})
	run := make(chan struct{})
	result := make(chan preexistingThreadContainmentResult)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		before := currentThreadSeccompFilterCount()
		close(ready)
		<-run
		result <- preexistingThreadContainmentResult{
			before:      before,
			after:       currentThreadSeccompFilterCount(),
			containment: inspectContainedThread(),
		}
	}()
	return ready, run, result
}

func runInheritedProcessContainmentProbe() int {
	if !inspectProcessContainment().valid() {
		return failure(os.Stderr, "inherited_process_containment_failed")
	}
	fmt.Fprintln(os.Stdout, "inherited-ok")
	return 0
}

func runProcessContainmentProbe(
	runThread chan struct{},
	threadResult chan preexistingThreadContainmentResult,
) int {
	close(runThread)
	if !(<-threadResult).valid() {
		return failure(os.Stderr, "process_containment_tsync_failed")
	}
	if !inspectProcessContainment().valid() {
		return failure(os.Stderr, "process_containment_probe_failed")
	}
	command := exec.Command(
		"/proc/self/exe",
		inheritedProcessContainmentProbeArgument,
	)
	command.Env = []string{}
	output, err := command.CombinedOutput()
	if err != nil || string(output) != "inherited-ok\n" {
		return failure(os.Stderr, "process_containment_inheritance_failed")
	}
	fmt.Fprintln(os.Stdout, processContainment)
	return 0
}

type finding struct {
	RuleID    string `json:"rule_id"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
}

type result struct {
	Type     string    `json:"type"`
	Ordinal  uint64    `json:"ordinal"`
	Bytes    uint64    `json:"bytes"`
	SHA256   string    `json:"sha256"`
	Findings []finding `json:"findings"`
}

type pathRule struct {
	RuleID          string `json:"rule_id"`
	Selector        string `json:"selector"`
	HasContentRegex bool   `json:"has_content_regex"`
}

type ready struct {
	Type                    string     `json:"type"`
	Protocol                string     `json:"protocol"`
	Version                 string     `json:"version"`
	ConfigSHA256            string     `json:"config_sha256"`
	RuleCount               int        `json:"rule_count"`
	PathRules               []pathRule `json:"path_rules"`
	RemovedContentPathRules []string   `json:"removed_content_path_rules"`
	PolicyBeforeSHA256      string     `json:"policy_before_sha256"`
	PolicyAfterSHA256       string     `json:"policy_after_sha256"`
	MaxTargetMegaBytes      int        `json:"max_target_megabytes"`
	IgnoreInlineAllow       bool       `json:"ignore_inline_allow"`
	Redact                  uint       `json:"redact"`
	ProcessContainment      string     `json:"process_containment"`
}

type summary struct {
	Type     string `json:"type"`
	Files    uint64 `json:"files"`
	Bytes    uint64 `json:"bytes"`
	Findings uint64 `json:"findings"`
}

var authorizedContentPaths = map[string]string{
	"freemius-secret-key":    `(?i)\.php$`,
	"hashicorp-tf-password":  `(?i)\.(?:tf|hcl)$`,
	"kubernetes-secret-yaml": `(?i)\.ya?ml$`,
	"nuget-config-password":  `(?i)nuget\.config$`,
}

const pkcs12Selector = `(?i)(?:^|\/)[^\/]+\.p(?:12|fx)$`

func canonicalAllowlistSets(input []*config.Allowlist) []*config.Allowlist {
	if input == nil {
		return nil
	}
	result := make([]*config.Allowlist, len(input))
	for index, allow := range input {
		if allow == nil {
			continue
		}
		copied := *allow
		if allow.Commits != nil {
			copied.Commits = append(make([]string, 0, len(allow.Commits)), allow.Commits...)
			sort.Strings(copied.Commits)
		}
		if allow.StopWords != nil {
			copied.StopWords = append(make([]string, 0, len(allow.StopWords)), allow.StopWords...)
			sort.Strings(copied.StopWords)
		}
		result[index] = &copied
	}
	return result
}

func policyDigest(cfg config.Config) (string, error) {
	cfg.Path = "" // Input config path is separately bound by its exact byte hash.
	// Upstream Validate deduplicates these semantic sets using map keys. Sort
	// private copies solely for hashing; never alter active detector policy.
	cfg.Allowlists = canonicalAllowlistSets(cfg.Allowlists)
	originalRules := cfg.Rules
	cfg.Rules = make(map[string]config.Rule, len(originalRules))
	for id, rule := range originalRules {
		rule.Allowlists = canonicalAllowlistSets(rule.Allowlists)
		cfg.Rules[id] = rule
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func strengthenContentRules(cfg *config.Config) ([]string, string) {
	// These four exact upstream selectors are reviewed policy. Change only the
	// selector prerequisite; all content/entropy/keyword/allowlist policy stays.
	for id, rule := range cfg.Rules {
		if rule.Path == nil {
			continue
		}
		if id == "pkcs12-file" && rule.Regex == nil && rule.Path.String() == pkcs12Selector {
			continue
		}
		expected, ok := authorizedContentPaths[id]
		if !ok || rule.Regex == nil || rule.Path.String() != expected {
			return nil, "unknown_path_rule"
		}
	}
	removed := make([]string, 0, len(authorizedContentPaths))
	for id, expected := range authorizedContentPaths {
		rule, ok := cfg.Rules[id]
		if !ok || rule.Path == nil || rule.Path.String() != expected || rule.Regex == nil {
			return nil, "missing_reviewed_content_path_rule"
		}
		rule.Path = nil
		cfg.Rules[id] = rule
		removed = append(removed, id)
	}
	sort.Strings(removed)
	return removed, ""
}

// Metadata is checked around the complete descriptor read, including ctime:
// same-size rewrites cannot be hidden by restoring the previous mtime.
func sameConfigStat(before, after os.FileInfo) bool {
	if !before.Mode().IsRegular() || !after.Mode().IsRegular() ||
		!os.SameFile(before, after) || before.Size() != after.Size() ||
		before.Mode() != after.Mode() || !before.ModTime().Equal(after.ModTime()) {
		return false
	}
	left, leftOK := before.Sys().(*syscall.Stat_t)
	right, rightOK := after.Sys().(*syscall.Stat_t)
	return leftOK && rightOK && left.Ctim == right.Ctim &&
		left.Nlink == right.Nlink && left.Uid == right.Uid && left.Gid == right.Gid
}

func readConfigData(file *os.File, readAll func(io.Reader) ([]byte, error)) ([]byte, string) {
	before, err := file.Stat()
	if err != nil || !before.Mode().IsRegular() {
		return nil, "config_not_regular"
	}
	position, err := file.Seek(0, io.SeekCurrent)
	if err != nil || position != 0 {
		return nil, "config_position_invalid"
	}
	// ReaderAt preserves the inherited open-file position. Read to actual EOF,
	// never just a caller-selected byte range or a configured size limit.
	data, err := readAll(io.NewSectionReader(file, 0, math.MaxInt64))
	if err != nil {
		return nil, "config_read_error"
	}
	after, err := file.Stat()
	if err != nil || !sameConfigStat(before, after) || int64(len(data)) != after.Size() {
		return nil, "config_changed"
	}
	return data, ""
}

func configuration(path string) (config.Config, ready, string) {
	before, err := os.Lstat(path)
	if err != nil || !before.Mode().IsRegular() {
		return config.Config{}, ready{}, "config_not_regular"
	}
	file, err := os.OpenFile(path, os.O_RDONLY|syscall.O_NOFOLLOW|syscall.O_NONBLOCK, 0)
	if err != nil {
		return config.Config{}, ready{}, "config_read_error"
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil || !sameConfigStat(before, opened) {
		return config.Config{}, ready{}, "config_changed"
	}
	data, code := readConfigData(file, io.ReadAll)
	if code != "" {
		return config.Config{}, ready{}, code
	}
	current, err := os.Lstat(path)
	if err != nil || !sameConfigStat(opened, current) {
		return config.Config{}, ready{}, "config_changed"
	}
	return parseConfiguration(data, path)
}

func configurationFD(fd int) (config.Config, ready, string) {
	if fd < 3 {
		return config.Config{}, ready{}, "config_descriptor_invalid"
	}
	duplicate, err := syscall.Dup(fd)
	if err != nil {
		return config.Config{}, ready{}, "config_descriptor_invalid"
	}
	syscall.CloseOnExec(duplicate)
	file := os.NewFile(uintptr(duplicate), "verified-config-descriptor")
	if file == nil {
		syscall.Close(duplicate)
		return config.Config{}, ready{}, "config_descriptor_invalid"
	}
	defer file.Close()
	data, code := readConfigData(file, io.ReadAll)
	if code != "" {
		return config.Config{}, ready{}, code
	}
	// This controlled label never receives an archive path and cannot match an
	// extensionless ordinal. Input bytes remain bound by the same ready hash.
	return parseConfiguration(data, "<verified-config-descriptor>")
}

func parseConfiguration(data []byte, path string) (config.Config, ready, string) {
	// A private Viper instance avoids environment/config auto-discovery. Translate
	// reads only its embedded default extension; external extension is rejected.
	parser := viper.New()
	parser.SetConfigType("toml")
	if err := parser.ReadConfig(bytes.NewReader(data)); err != nil {
		return config.Config{}, ready{}, "config_parse_error"
	}
	var input config.ViperConfig
	if err := parser.UnmarshalExact(&input); err != nil {
		return config.Config{}, ready{}, "config_schema_error"
	}
	if !input.Extend.UseDefault || input.Extend.Path != "" || input.Extend.URL != "" || len(input.Extend.DisabledRules) != 0 {
		return config.Config{}, ready{}, "config_extension_rejected"
	}
	parsed, err := input.Translate()
	if err != nil {
		return config.Config{}, ready{}, "config_translate_error"
	}
	if len(parsed.Rules) == 0 {
		return config.Config{}, ready{}, "config_empty_rules"
	}
	parsed.Path = path
	paths := make([]pathRule, 0)
	for id, rule := range parsed.Rules {
		if rule.Path != nil {
			paths = append(paths, pathRule{id, rule.Path.String(), rule.Regex != nil})
		}
	}
	sort.Slice(paths, func(i, j int) bool { return paths[i].RuleID < paths[j].RuleID })
	beforePolicy, err := policyDigest(parsed)
	if err != nil {
		return config.Config{}, ready{}, "policy_digest_error"
	}
	removed, policyError := strengthenContentRules(&parsed)
	if policyError != "" {
		return config.Config{}, ready{}, policyError
	}
	afterPolicy, err := policyDigest(parsed)
	if err != nil {
		return config.Config{}, ready{}, "policy_digest_error"
	}
	digest := sha256.Sum256(data)
	return parsed, ready{Type: "ready", Protocol: "whole-file-gitleaks.v1", Version: detectorVersion,
		ConfigSHA256: hex.EncodeToString(digest[:]), RuleCount: len(parsed.Rules), PathRules: paths,
		RemovedContentPathRules: removed, PolicyBeforeSHA256: beforePolicy, PolicyAfterSHA256: afterPolicy,
		MaxTargetMegaBytes: 0, IgnoreInlineAllow: true, Redact: 100,
		ProcessContainment: processContainment}, ""
}

func failure(stderr io.Writer, code string) int {
	// Only controlled type codes are emitted; never interpolate inputs/errors.
	_ = json.NewEncoder(stderr).Encode(map[string]string{"error": code})
	return 2
}

// inFlightByteBudget bounds the payload bytes admitted to the detection pool at
// one time. Detection allocates roughly three copies of a record (the string
// conversion, the lowercased prefilter copy, and match state), so bounding
// admitted bytes bounds the detector heap independently of archive size. A
// record larger than the entire budget is admitted alone rather than refused:
// coverage never depends on a size threshold.
const inFlightByteBudget = 512 << 20

// inFlightJobsPerWorker bounds outstanding records per worker so that a stream
// of empty or tiny records cannot accumulate without limit under the byte
// budget, which such records barely consume.
const inFlightJobsPerWorker = 4

// reclaimIntervalBytes is how many completed payload bytes trigger one forced
// heap reclamation. Reclaiming after every record measured 0.54% of wall time
// on a representative corpus and would serialise the pool on a stop-the-world
// pause per record; reclaiming per budget-sized batch keeps the resident set
// bounded at a fraction of that cost.
const reclaimIntervalBytes = inFlightByteBudget

// byteBudget is a weighted semaphore over payload bytes admitted for detection.
// A reservation is taken before the payload is allocated, so the budget bounds
// the bytes this process allocates for records rather than only the bytes it has
// already read.
type byteBudget struct {
	mutex     sync.Mutex
	returned  *sync.Cond
	available int64
	capacity  int64
	cancelled bool
}

// newByteBudget builds a budget holding capacity bytes.
//
// Args:
//
//	capacity: Maximum payload bytes admitted for detection at one time.
//
// Returns:
//
//	A budget with its whole capacity available.
func newByteBudget(capacity int64) *byteBudget {
	budget := &byteBudget{available: capacity, capacity: capacity}
	budget.returned = sync.NewCond(&budget.mutex)
	return budget
}

// acquire reserves room for one record, blocking until it is available.
//
// Args:
//
//	want: Payload bytes the record is about to allocate.
//
// Returns:
//
//	The reserved amount, clamped to the whole capacity so an oversized record
//	runs alone instead of deadlocking, and whether the reservation was granted.
//	Release exactly the returned amount. A cancelled budget grants nothing, so a
//	reader waiting for room stops instead of outliving a failed pipeline.
func (b *byteBudget) acquire(want int64) (int64, bool) {
	if want > b.capacity {
		want = b.capacity
	}
	b.mutex.Lock()
	defer b.mutex.Unlock()
	for b.available < want && !b.cancelled {
		b.returned.Wait()
	}
	if b.cancelled {
		return 0, false
	}
	b.available -= want
	return want, true
}

// cancel makes every current and future acquire return without a reservation.
//
// Args:
//
//	None.
//
// Returns:
//
//	None.
func (b *byteBudget) cancel() {
	b.mutex.Lock()
	b.cancelled = true
	b.mutex.Unlock()
	b.returned.Broadcast()
}

// release returns a previously reserved amount to the budget.
//
// Args:
//
//	amount: The exact value a matching acquire returned.
//
// Returns:
//
//	None.
func (b *byteBudget) release(amount int64) {
	b.mutex.Lock()
	b.available += amount
	b.mutex.Unlock()
	b.returned.Broadcast()
}

// scanJob is one complete record moving through the detection pool. The
// collector emits jobs in ordinal order, so output never depends on the number
// of workers or on which worker finished first.
type scanJob struct {
	ordinal  uint64
	length   uint64
	payload  []byte
	reserved int64
	digest   string
	findings []finding
	panicked bool
	failure  string
	done     chan struct{}
}

// recordPath builds the controlled path a record is detected under.
//
// Args:
//
//	ordinal: The record's one-based position in the stream.
//
// Returns:
//
//	A fixed-width extensionless label that carries no archive path.
func recordPath(ordinal uint64) string {
	return fmt.Sprintf("record-%020d", ordinal)
}

// controlledPathCode reports the failure code when a controlled record label
// would collide with the configuration path or activate a path allowlist.
//
// Args:
//
//	detector: The configured detector whose policy is checked.
//	ordinal: The record's one-based position in the stream.
//
// Returns:
//
//	A controlled failure code, or "" when the label is safe to detect under.
func controlledPathCode(detector *detect.Detector, ordinal uint64) string {
	controlledPath := recordPath(ordinal)
	if controlledPath == detector.Config.Path {
		return "controlled_path_matches_config"
	}
	for _, allow := range detector.Config.Allowlists {
		if allow.PathAllowed(controlledPath) {
			return "controlled_path_allowlisted"
		}
	}
	for _, rule := range detector.Config.Rules {
		for _, allow := range rule.Allowlists {
			if allow.PathAllowed(controlledPath) {
				return "controlled_path_allowlisted"
			}
		}
	}
	return ""
}

// sortFindings orders findings so identical detections serialise identically.
//
// Args:
//
//	findings: Findings for one record, reordered in place.
//
// Returns:
//
//	None.
func sortFindings(findings []finding) {
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].RuleID != findings[j].RuleID {
			return findings[i].RuleID < findings[j].RuleID
		}
		if findings[i].StartLine != findings[j].StartLine {
			return findings[i].StartLine < findings[j].StartLine
		}
		return findings[i].EndLine < findings[j].EndLine
	})
}

// scanRaw applies Gitleaks once to one complete immutable string.
func scanRaw(raw string, ordinal uint64, detector *detect.Detector) ([]finding, string) {
	matches := detector.Detect(detect.Fragment{Raw: raw, FilePath: recordPath(ordinal)})
	if len(matches) > maxRecordFindings {
		return nil, "record_finding_limit"
	}
	findings := make([]finding, 0, len(matches))
	for _, match := range matches {
		findings = append(findings, finding{match.RuleID, match.StartLine, match.EndLine})
	}
	sortFindings(findings)
	return findings, ""
}

// detectRecord hashes and scans one complete record, then releases its budget.
// A panic is contained and reported on the job so the collector can fail closed
// rather than losing the whole process.
//
// Args:
//
//	job: The record to scan; its results are written back onto the job.
//	detector: The shared configured detector, which upstream invokes
//	  concurrently and which holds no mutable per-scan state.
//	budget: The budget holding this job's reservation.
//
// Returns:
//
//	None.
func detectRecord(job *scanJob, detector *detect.Detector, budget *byteBudget) {
	defer close(job.done)
	defer func() {
		if recover() != nil {
			job.panicked = true
		}
	}()
	defer budget.release(job.reserved)
	digest := sha256.Sum256(job.payload)
	job.digest = hex.EncodeToString(digest[:])
	// One call, one complete file. No MIME decision, source chunker, overlap,
	// archive traversal, baseline, ignore file, or AddFinding accumulation.
	findings, code := scanRaw(string(job.payload), job.ordinal, detector)
	job.payload = nil
	job.failure = code
	job.findings = findings
}

// startDetectionPool runs the detection workers until dispatch closes or the
// pipeline aborts. Workers watch abort directly rather than waiting for the
// reader to close dispatch, because the reader may be parked in a read on an
// input the caller has not ended.
//
// Args:
//
//	workers: Number of concurrent detections.
//	dispatch: Jobs to scan; closed by the reader at end of input.
//	detector: The shared configured detector.
//	budget: The budget each finished job releases into.
//	abort: Closed once the collector has already failed.
//
// Returns:
//
//	A group that completes once every worker has stopped, which is the point at
//	which no detection is still running.
func startDetectionPool(workers int, dispatch <-chan *scanJob, detector *detect.Detector,
	budget *byteBudget, abort <-chan struct{}) *sync.WaitGroup {
	var pool sync.WaitGroup
	for index := 0; index < workers; index++ {
		pool.Add(1)
		go func() {
			defer pool.Done()
			for {
				select {
				case job, open := <-dispatch:
					if !open {
						return
					}
					detectRecord(job, detector, budget)
				case <-abort:
					return
				}
			}
		}()
	}
	return &pool
}

// readLength reads one record's framing header.
//
// Args:
//
//	input: The framed record stream, positioned at a header.
//	files: How many records have already been admitted.
//	consumed: How many payload bytes have already been admitted.
//
// Returns:
//
//	The record's payload length, or -1 with "" at clean end of input, and the
//	controlled failure code when the framing itself is unusable.
func readLength(input io.Reader, files, consumed uint64) (int64, string) {
	var header [8]byte
	count, err := io.ReadFull(input, header[:])
	if err == io.EOF && count == 0 {
		return -1, ""
	}
	if err != nil {
		return -1, "truncated_header"
	}
	length := binary.BigEndian.Uint64(header[:])
	if length > maxRecordBytes {
		return -1, "record_size_limit"
	}
	if length > uint64(int(^uint(0)>>1)) || consumed > math.MaxUint64-length || files == math.MaxUint64 {
		return -1, "length_overflow"
	}
	return int64(length), ""
}

// admitOne reserves room for one record, reads it, and checks its path policy.
// The reservation is taken before the payload is allocated and is released again
// on every path that does not produce a job, so admitted bytes bound the payload
// bytes this process holds rather than trailing them by one whole record.
//
// Args:
//
//	input: The framed record stream, positioned at a payload.
//	detector: The shared configured detector, consulted for path policy.
//	budget: The byte budget gating admission.
//	ordinal: This record's one-based position in the stream.
//	length: This record's exact payload length.
//
// Returns:
//
//	The admitted job, or nil with the controlled failure code, which is "" when
//	the budget was cancelled because the pipeline had already failed.
func admitOne(input io.Reader, detector *detect.Detector, budget *byteBudget,
	ordinal uint64, length int64) (*scanJob, string) {
	reserved, granted := budget.acquire(length)
	if !granted {
		return nil, ""
	}
	payload := make([]byte, int(length))
	if _, err := io.ReadFull(input, payload); err != nil {
		budget.release(reserved)
		return nil, "truncated_payload"
	}
	if code := controlledPathCode(detector, ordinal); code != "" {
		budget.release(reserved)
		return nil, code
	}
	return &scanJob{
		ordinal:  ordinal,
		length:   uint64(length),
		payload:  payload,
		reserved: reserved,
		done:     make(chan struct{}),
	}, ""
}

// admitRecords reads framed records in stream order and hands them to the pool.
// Records admitted before a failure are still emitted, which keeps output on
// every failure path identical to scanning one record at a time.
//
// Args:
//
//	input: The framed record stream.
//	detector: The shared configured detector, consulted for path policy.
//	budget: The byte budget gating admission, reserved before each allocation
//	  and released again on every path that does not hand the job to a worker.
//	dispatch: Channel the workers consume.
//	ordered: Channel the collector consumes, written in ordinal order.
//	abort: Closed by the caller when the collector has already failed.
//
// Returns:
//
//	The controlled failure code that stopped the stream, or "" at clean EOF or
//	when the pipeline aborted.
func admitRecordsFrom(input io.Reader, detector *detect.Detector, budget *byteBudget,
	dispatch chan<- *scanJob, ordered chan<- *scanJob, abort <-chan struct{}, ordinalBase uint64) string {
	var files, consumed uint64
	for {
		if ordinalBase > math.MaxUint64-files {
			return "length_overflow"
		}
		length, code := readLength(input, ordinalBase+files, consumed)
		if code != "" || length < 0 {
			return code
		}
		files++
		consumed += uint64(length)
		job, code := admitOne(input, detector, budget, ordinalBase+files, length)
		if job == nil {
			return code
		}
		select {
		case ordered <- job:
		case <-abort:
			budget.release(job.reserved)
			return ""
		}
		select {
		case dispatch <- job:
		case <-abort:
			// The job reached ordered but no worker took it, so this goroutine
			// still owns the reservation.
			budget.release(job.reserved)
			return ""
		}
	}
}

func admitRecords(input io.Reader, detector *detect.Detector, budget *byteBudget,
	dispatch chan<- *scanJob, ordered chan<- *scanJob, abort <-chan struct{}) string {
	return admitRecordsFrom(input, detector, budget, dispatch, ordered, abort, 0)
}

// emitRecords writes one result per record in ordinal order.
//
// Args:
//
//	ordered: Jobs in stream order; closed by the reader.
//	emit: Serialises one protocol value and flushes it.
//
// Returns:
//
//	The running summary and the controlled failure code that stopped emission,
//	or "" when every admitted record was emitted.
func emitRecords(ordered <-chan *scanJob, emit func(any) bool) (summary, string) {
	totals := summary{Type: "summary"}
	var sinceReclaim int64
	for job := range ordered {
		<-job.done
		if job.panicked {
			return totals, "internal_panic"
		}
		if job.failure != "" {
			return totals, job.failure
		}
		if totals.Findings > math.MaxUint64-uint64(len(job.findings)) {
			return totals, "finding_count_overflow"
		}
		totals.Files++
		totals.Bytes += job.length
		totals.Findings += uint64(len(job.findings))
		if !emit(result{Type: "result", Ordinal: job.ordinal, Bytes: job.length, SHA256: job.digest, Findings: job.findings}) {
			return totals, "output_error"
		}
		sinceReclaim += int64(job.length)
		if sinceReclaim >= reclaimIntervalBytes {
			sinceReclaim = 0
			runtime.GC()
		}
	}
	return totals, ""
}

// runPipeline scans the whole stream through the detection pool.
//
// Detection of distinct records overlaps, but emission is strictly ordinal, so
// the protocol bytes are identical to scanning one record at a time. Worker
// count follows GOMAXPROCS, which in production is the schedulable CPU count:
// the scanner starts this helper with only PATH in its environment, so a shell
// GOMAXPROCS reaches a hand-run helper but never the one the scanner owns.
//
// Args:
//
//	input: The framed record stream.
//	detector: The shared configured detector.
//	emit: Serialises one protocol value and flushes it.
//
// Returns:
//
//	The summary and the controlled failure code, or "" when the stream ended
//	cleanly and every record was emitted.
func detectionWorkers() int {
	workers := runtime.GOMAXPROCS(0)
	if workers > maxDetectionWorkers {
		return maxDetectionWorkers
	}
	return workers
}

func runPipelineFrom(input io.Reader, detector *detect.Detector, emit func(any) bool,
	ordinalBase uint64) (summary, string) {
	workers := detectionWorkers()
	budget := newByteBudget(inFlightByteBudget)
	dispatch := make(chan *scanJob, workers)
	ordered := make(chan *scanJob, workers*inFlightJobsPerWorker)
	abort := make(chan struct{})
	pool := startDetectionPool(workers, dispatch, detector, budget, abort)
	admitted := make(chan string, 1)
	go func() {
		code := admitRecordsFrom(input, detector, budget, dispatch, ordered, abort, ordinalBase)
		close(dispatch)
		close(ordered)
		admitted <- code
	}()
	totals, emitCode := emitRecords(ordered, emit)
	if emitCode != "" {
		// Stop every worker and free the reader from any wait this process
		// controls, then report the failure. The reader may still be parked in a
		// read on input only the caller can end; waiting for it here would make a
		// contained failure depend on the caller sending more bytes or closing.
		close(abort)
		budget.cancel()
		pool.Wait()
		return totals, emitCode
	}
	pool.Wait()
	return totals, <-admitted
}

func runPipeline(input io.Reader, detector *detect.Detector, emit func(any) bool) (summary, string) {
	return runPipelineFrom(input, detector, emit, 0)
}

func processFrom(input io.Reader, output, stderr io.Writer, detector *detect.Detector,
	info ready, ordinalBase uint64) (exit int) {
	defer func() {
		if recover() != nil {
			exit = failure(stderr, "internal_panic")
		}
	}()
	writer := bufio.NewWriter(output)
	encoder := json.NewEncoder(writer)
	emit := func(value any) bool { return encoder.Encode(value) == nil && writer.Flush() == nil }
	if !emit(info) {
		return failure(stderr, "output_error")
	}
	totals, code := runPipelineFrom(input, detector, emit, ordinalBase)
	if code != "" {
		return failure(stderr, code)
	}
	if !emit(totals) {
		return failure(stderr, "output_error")
	}
	if totals.Findings != 0 {
		return 1
	}
	return 0
}

func process(input io.Reader, output, stderr io.Writer, detector *detect.Detector, info ready) int {
	return processFrom(input, output, stderr, detector, info, 0)
}

func mapPrivateRecord(fd int, length uint64) (*os.File, []byte, os.FileInfo, string) {
	if length > completeRecordLimit {
		return nil, nil, nil, "complete_record_limit"
	}
	if fd < 3 || length == 0 || length > uint64(int(^uint(0)>>1)) {
		return nil, nil, nil, "record_descriptor_invalid"
	}
	file := os.NewFile(uintptr(fd), "private-oversized-record")
	if file == nil {
		return nil, nil, nil, "record_descriptor_invalid"
	}
	before, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, nil, nil, "record_descriptor_invalid"
	}
	system, ok := before.Sys().(*syscall.Stat_t)
	flags, _, flagErr := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), syscall.F_GETFL, 0)
	if !ok || !before.Mode().IsRegular() || before.Size() < 0 ||
		uint64(before.Size()) != length || before.Mode().Perm() != 0400 ||
		before.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 ||
		system.Uid != uint32(os.Geteuid()) || system.Nlink != 0 ||
		flagErr != 0 || int(flags)&syscall.O_ACCMODE != syscall.O_RDONLY {
		file.Close()
		return nil, nil, nil, "record_descriptor_invalid"
	}
	payload, err := syscall.Mmap(fd, 0, int(length), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, nil, nil, "record_mapping_failed"
	}
	return file, payload, before, ""
}

func detectMappedRecord(payload []byte, detector *detect.Detector, ordinal uint64) (found []finding, code string) {
	defer func() {
		if recover() != nil {
			found = nil
			code = "internal_panic"
		}
	}()
	raw := unsafe.String(unsafe.SliceData(payload), len(payload))
	found, code = scanRaw(raw, ordinal, detector)
	raw = ""
	runtime.KeepAlive(payload)
	return found, code
}

func scanMappedRecord(detector *detect.Detector, recordFD int,
	length, ordinal uint64) (result, summary, string) {
	if ordinal == 0 {
		return result{}, summary{}, "record_ordinal_invalid"
	}
	if code := controlledPathCode(detector, ordinal); code != "" {
		return result{}, summary{}, code
	}
	file, payload, before, code := mapPrivateRecord(recordFD, length)
	if code != "" {
		return result{}, summary{}, code
	}
	first := sha256.Sum256(payload)
	findings, scanCode := mappedRecordDetect(payload, detector, ordinal)
	second := sha256.Sum256(payload)
	after, statError := mappedRecordStat(file)
	mutation := statError != nil || first != second ||
		(statError == nil && !sameConfigStat(before, after))
	unmapError := mappedRecordUnmap(payload)
	closeError := mappedRecordClose(file)
	if scanCode != "" {
		return result{}, summary{}, scanCode
	}
	if mutation {
		return result{}, summary{}, "record_changed"
	}
	if unmapError != nil {
		return result{}, summary{}, "record_unmap_failed"
	}
	if closeError != nil {
		return result{}, summary{}, "record_close_failed"
	}
	digest := hex.EncodeToString(first[:])
	record := result{Type: "result", Ordinal: ordinal, Bytes: length, SHA256: digest, Findings: findings}
	totals := summary{Type: "summary", Files: 1, Bytes: length, Findings: uint64(len(findings))}
	return record, totals, ""
}

func processMapped(output, stderr io.Writer, detector *detect.Detector, info ready,
	recordFD int, length, ordinal uint64) (exit int) {
	defer func() {
		if recover() != nil {
			exit = failure(stderr, "internal_panic")
		}
	}()
	writer := bufio.NewWriter(output)
	encoder := json.NewEncoder(writer)
	emit := func(value any) bool { return encoder.Encode(value) == nil && writer.Flush() == nil }
	if !emit(info) {
		return failure(stderr, "output_error")
	}
	record, totals, code := scanMappedRecord(detector, recordFD, length, ordinal)
	if code != "" {
		return failure(stderr, code)
	}
	if !emit(record) || !emit(totals) {
		return failure(stderr, "output_error")
	}
	if totals.Findings != 0 {
		return 1
	}
	return 0
}

type scannerOptions struct {
	configPath    string
	configFD      int
	recordFD      int
	recordLength  uint64
	recordOrdinal uint64
	ordinalBase   uint64
	configByFD    bool
	mapped        bool
}

func validScannerOptions(supplied map[string]bool, options scannerOptions) bool {
	mappedComplete := supplied["record-fd"] &&
		supplied["record-length"] && supplied["record-ordinal"]
	return supplied["config"] != supplied["config-fd"] &&
		(!supplied["config"] || options.configPath != "") &&
		(!supplied["config-fd"] || options.configFD >= 3) &&
		(!options.mapped || (mappedComplete && options.configByFD &&
			!supplied["ordinal-base"] && options.recordFD >= 3 &&
			options.recordFD != options.configFD && options.recordLength != 0 &&
			options.recordOrdinal != 0))
}

func parseScannerOptions(args []string) (scannerOptions, bool) {
	flags := flag.NewFlagSet("whole-file-scanner", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	path := flags.String("config", "", "Trusted repository config")
	fd := flags.Int("config-fd", -1, "Inherited verified regular-file config descriptor")
	recordFD := flags.Int("record-fd", -1, "Inherited private oversized-record descriptor")
	recordLength := flags.Uint64("record-length", 0, "Exact oversized-record byte count")
	recordOrdinal := flags.Uint64("record-ordinal", 0, "Global one-based record ordinal")
	ordinalBase := flags.Uint64("ordinal-base", 0, "Global ordinal preceding this stream")
	if flags.Parse(args) != nil || flags.NArg() != 0 {
		return scannerOptions{}, false
	}
	supplied := map[string]bool{}
	flags.Visit(func(item *flag.Flag) { supplied[item.Name] = true })
	mapped := supplied["record-fd"] || supplied["record-length"] || supplied["record-ordinal"]
	options := scannerOptions{
		configPath:    *path,
		configFD:      *fd,
		recordFD:      *recordFD,
		recordLength:  *recordLength,
		recordOrdinal: *recordOrdinal,
		ordinalBase:   *ordinalBase,
		configByFD:    supplied["config-fd"],
		mapped:        mapped,
	}
	return options, validScannerOptions(supplied, options)
}

func run(args []string, input io.Reader, output, stderr io.Writer) (exit int) {
	defer func() {
		if recover() != nil {
			exit = failure(stderr, "configuration_panic")
		}
	}()
	log.SetOutput(io.Discard)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	options, valid := parseScannerOptions(args)
	if !valid {
		return failure(stderr, "invalid_arguments")
	}
	var parsed config.Config
	var info ready
	var code string
	if options.configByFD {
		parsed, info, code = configurationFD(options.configFD)
	} else {
		parsed, info, code = configuration(options.configPath)
	}
	if code != "" {
		return failure(stderr, code)
	}
	detector := detect.NewDetector(parsed)
	detector.MaxTargetMegaBytes = 0
	detector.IgnoreGitleaksAllow = true
	detector.Redact = 100
	detector.Verbose = false
	if options.mapped {
		return processMapped(output, stderr, detector, info,
			options.recordFD, options.recordLength, options.recordOrdinal)
	}
	return processFrom(input, output, stderr, detector, info, options.ordinalBase)
}

func boundedAddressSpace(requested uint64, inherited syscall.Rlimit) uint64 {
	effective := requested
	for _, limit := range []uint64{inherited.Cur, inherited.Max} {
		if limit != math.MaxUint64 && limit < effective {
			effective = limit
		}
	}
	return effective
}

func applyAddressSpaceLimit(requested uint64) (uint64, error) {
	var inherited syscall.Rlimit
	if err := syscall.Getrlimit(syscall.RLIMIT_AS, &inherited); err != nil {
		return 0, err
	}
	effective := boundedAddressSpace(requested, inherited)
	next := syscall.Rlimit{Cur: effective, Max: effective}
	if err := syscall.Setrlimit(syscall.RLIMIT_AS, &next); err != nil {
		return 0, err
	}
	debug.SetMemoryLimit(int64(effective))
	return effective, nil
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == inheritedProcessContainmentProbeArgument {
		os.Exit(runInheritedProcessContainmentProbe())
	}
	probe := len(os.Args) == 2 && os.Args[1] == processContainmentProbeArgument
	var runThread chan struct{}
	var threadResult chan preexistingThreadContainmentResult
	if probe {
		ready, run, result := startPreexistingThreadProbe()
		<-ready
		runThread, threadResult = run, result
	}
	if err := installProcessContainment(); err != nil {
		os.Exit(failure(os.Stderr, "process_containment_failed"))
	}
	if probe {
		os.Exit(runProcessContainmentProbe(runThread, threadResult))
	}
	if _, err := applyAddressSpaceLimit(maxMemoryBytes); err != nil {
		os.Exit(failure(os.Stderr, "memory_limit_failed"))
	}
	verified := false
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dependency := range info.Deps {
			if dependency.Path == "github.com/zricethezav/gitleaks/v8" && dependency.Version == "v"+detectorVersion && dependency.Replace == nil {
				verified = true
			}
		}
	}
	if !verified {
		os.Exit(failure(os.Stderr, "detector_version_mismatch"))
	}
	os.Exit(run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}
