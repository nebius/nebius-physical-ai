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
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"syscall"

	"github.com/rs/zerolog"
	"github.com/spf13/viper"
	"github.com/zricethezav/gitleaks/v8/config"
	"github.com/zricethezav/gitleaks/v8/detect"
)

const detectorVersion = "8.28.0"

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
		MaxTargetMegaBytes: 0, IgnoreInlineAllow: true, Redact: 100}, ""
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
	matches := detector.Detect(detect.Fragment{Raw: string(job.payload), FilePath: recordPath(job.ordinal)})
	job.payload = nil
	findings := make([]finding, 0, len(matches))
	for _, match := range matches {
		findings = append(findings, finding{match.RuleID, match.StartLine, match.EndLine})
	}
	sortFindings(findings)
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
func admitRecords(input io.Reader, detector *detect.Detector, budget *byteBudget,
	dispatch chan<- *scanJob, ordered chan<- *scanJob, abort <-chan struct{}) string {
	var files, consumed uint64
	for {
		length, code := readLength(input, files, consumed)
		if code != "" || length < 0 {
			return code
		}
		files++
		consumed += uint64(length)
		job, code := admitOne(input, detector, budget, files, length)
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
func runPipeline(input io.Reader, detector *detect.Detector, emit func(any) bool) (summary, string) {
	workers := runtime.GOMAXPROCS(0)
	budget := newByteBudget(inFlightByteBudget)
	dispatch := make(chan *scanJob, workers)
	ordered := make(chan *scanJob, workers*inFlightJobsPerWorker)
	abort := make(chan struct{})
	pool := startDetectionPool(workers, dispatch, detector, budget, abort)
	admitted := make(chan string, 1)
	go func() {
		code := admitRecords(input, detector, budget, dispatch, ordered, abort)
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

func process(input io.Reader, output, stderr io.Writer, detector *detect.Detector, info ready) (exit int) {
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
	totals, code := runPipeline(input, detector, emit)
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

func run(args []string, input io.Reader, output, stderr io.Writer) (exit int) {
	defer func() {
		if recover() != nil {
			exit = failure(stderr, "configuration_panic")
		}
	}()
	log.SetOutput(io.Discard)
	zerolog.SetGlobalLevel(zerolog.Disabled)
	flags := flag.NewFlagSet("whole-file-scanner", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	path := flags.String("config", "", "Trusted repository config")
	fd := flags.Int("config-fd", -1, "Inherited verified regular-file config descriptor")
	if flags.Parse(args) != nil || flags.NArg() != 0 {
		return failure(stderr, "invalid_arguments")
	}
	supplied := map[string]bool{}
	flags.Visit(func(item *flag.Flag) { supplied[item.Name] = true })
	if supplied["config"] == supplied["config-fd"] ||
		(supplied["config"] && *path == "") || (supplied["config-fd"] && *fd < 3) {
		return failure(stderr, "invalid_arguments")
	}
	var parsed config.Config
	var info ready
	var code string
	if supplied["config-fd"] {
		parsed, info, code = configurationFD(*fd)
	} else {
		parsed, info, code = configuration(*path)
	}
	if code != "" {
		return failure(stderr, code)
	}
	detector := detect.NewDetector(parsed)
	detector.MaxTargetMegaBytes = 0
	detector.IgnoreGitleaksAllow = true
	detector.Redact = 100
	detector.Verbose = false
	return process(input, output, stderr, detector, info)
}

func main() {
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
