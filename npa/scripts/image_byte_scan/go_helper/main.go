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
	totals := summary{Type: "summary"}
	for {
		var header [8]byte
		count, err := io.ReadFull(input, header[:])
		if err == io.EOF && count == 0 {
			if !emit(totals) {
				return failure(stderr, "output_error")
			}
			if totals.Findings != 0 {
				return 1
			}
			return 0
		}
		if err != nil {
			return failure(stderr, "truncated_header")
		}
		length := binary.BigEndian.Uint64(header[:])
		if length > uint64(int(^uint(0)>>1)) || totals.Bytes > math.MaxUint64-length || totals.Files == math.MaxUint64 {
			return failure(stderr, "length_overflow")
		}
		fragment := make([]byte, int(length))
		if _, err := io.ReadFull(input, fragment); err != nil {
			return failure(stderr, "truncated_payload")
		}
		ordinal := totals.Files + 1
		digest := sha256.Sum256(fragment)
		// One call, one complete file. No MIME decision, source chunker, overlap,
		// archive traversal, baseline, ignore file, or AddFinding accumulation.
		controlledPath := fmt.Sprintf("record-%020d", ordinal)
		if controlledPath == detector.Config.Path {
			return failure(stderr, "controlled_path_matches_config")
		}
		for _, allow := range detector.Config.Allowlists {
			if allow.PathAllowed(controlledPath) {
				return failure(stderr, "controlled_path_allowlisted")
			}
		}
		for _, rule := range detector.Config.Rules {
			for _, allow := range rule.Allowlists {
				if allow.PathAllowed(controlledPath) {
					return failure(stderr, "controlled_path_allowlisted")
				}
			}
		}
		matches := detector.Detect(detect.Fragment{Raw: string(fragment), FilePath: controlledPath})
		findings := make([]finding, 0, len(matches))
		for _, match := range matches {
			findings = append(findings, finding{match.RuleID, match.StartLine, match.EndLine})
		}
		sort.Slice(findings, func(i, j int) bool {
			if findings[i].RuleID != findings[j].RuleID {
				return findings[i].RuleID < findings[j].RuleID
			}
			if findings[i].StartLine != findings[j].StartLine {
				return findings[i].StartLine < findings[j].StartLine
			}
			return findings[i].EndLine < findings[j].EndLine
		})
		if totals.Findings > math.MaxUint64-uint64(len(findings)) {
			return failure(stderr, "finding_count_overflow")
		}
		totals.Files++
		totals.Bytes += length
		totals.Findings += uint64(len(findings))
		if !emit(result{Type: "result", Ordinal: ordinal, Bytes: length, SHA256: hex.EncodeToString(digest[:]), Findings: findings}) {
			return failure(stderr, "output_error")
		}
		fragment = nil
		matches = nil
		findings = nil
		runtime.GC()
	}
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
