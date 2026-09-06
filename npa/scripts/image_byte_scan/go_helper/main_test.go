package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/zricethezav/gitleaks/v8/config"
	"github.com/zricethezav/gitleaks/v8/detect"
	"github.com/zricethezav/gitleaks/v8/regexp"
)

var fixtureConfig config.Config
var fixtureReady ready

func TestMain(m *testing.M) {
	zerolog.SetGlobalLevel(zerolog.Disabled)
	var code string
	fixtureConfig, fixtureReady, code = configuration(configFixturePath())
	if code != "" {
		os.Stderr.WriteString("test_configuration_failed\n")
		os.Exit(2)
	}
	os.Exit(m.Run())
}

func scanner() *detect.Detector {
	d := detect.NewDetector(fixtureConfig)
	d.MaxTargetMegaBytes = 0
	d.IgnoreGitleaksAllow = true
	d.Redact = 100
	return d
}

func framed(values ...[]byte) []byte {
	var output bytes.Buffer
	for _, value := range values {
		var header [8]byte
		binary.BigEndian.PutUint64(header[:], uint64(len(value)))
		output.Write(header[:])
		output.Write(value)
	}
	return output.Bytes()
}

func protocol(t *testing.T, payload []byte) (int, []map[string]any, string) {
	t.Helper()
	var output, errors bytes.Buffer
	exit := process(bytes.NewReader(payload), &output, &errors, scanner(), fixtureReady)
	rows := make([]map[string]any, 0)
	decoder := json.NewDecoder(&output)
	for {
		var row map[string]any
		if err := decoder.Decode(&row); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal("invalid output JSON")
		}
		rows = append(rows, row)
	}
	return exit, rows, errors.String()
}

func syntheticPAT() string { return "gh" + "p_" + "aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hI6" }

func syntheticJWT() string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"HS256","typ":"JWT"}`))
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"synthetic":"` + strings.Repeat("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6000) + `"}`))
	signature := base64.RawURLEncoding.EncodeToString([]byte("synthetic-not-cryptographically-valid-signature"))
	return header + "." + payload + "." + signature
}

func syntheticPrivateKey() string {
	return "-----BEGIN " + "PRIVATE KEY-----\n" + strings.Repeat("aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0eF3hI6+/\n", 6000) + "-----END " + "PRIVATE KEY-----\n"
}

func hasRule(rows []map[string]any, id string) bool {
	for _, row := range rows {
		matches, ok := row["findings"].([]any)
		if !ok {
			continue
		}
		for _, match := range matches {
			if match.(map[string]any)["rule_id"] == id {
				return true
			}
		}
	}
	return false
}

func TestWholeFileAdversarialPayloads(t *testing.T) {
	cases := []struct {
		name, rule string
		data       []byte
	}{
		{"binary", "github-pat", append(append([]byte{0, 255, 128, 0}, []byte(syntheticPAT())...), 0, 255)},
		{"boundary", "github-pat", []byte(strings.Repeat("\x00", 99990) + syntheticPAT() + strings.Repeat("\x00", 99990))},
		{"unbounded_jwt", "jwt", []byte(strings.Repeat("\x00", 80000) + syntheticJWT() + "\n")},
		{"multiline_private_key", "private-key", []byte(syntheticPrivateKey())},
		{"inline_allow_ignored", "github-pat", []byte(syntheticPAT() + " # gitleaks:allow\n")},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			exit, rows, errors := protocol(t, framed(test.data))
			if exit != 1 || errors != "" || !hasRule(rows, test.rule) {
				t.Fatalf("detection failed: exit=%d expected_rule=%s", exit, test.rule)
			}
			encoded, _ := json.Marshal(rows)
			if bytes.Contains(encoded, test.data) || bytes.Contains(encoded, []byte(syntheticPAT())) {
				t.Fatal("raw fragment appeared in protocol")
			}
			if rows[1]["bytes"] != float64(len(test.data)) || rows[2]["bytes"] != float64(len(test.data)) {
				t.Fatal("byte coverage mismatch")
			}
			for _, item := range rows[1]["findings"].([]any) {
				match := item.(map[string]any)
				if len(match) != 3 || match["rule_id"] == nil || match["start_line"] == nil || match["end_line"] == nil {
					t.Fatal("unexpected finding disclosure fields")
				}
			}
		})
	}
}

func TestEmptyAndMultipleRecords(t *testing.T) {
	exit, rows, errors := protocol(t, framed(nil, []byte("ordinary data"), []byte(syntheticPAT()), []byte("final plain data")))
	if exit != 1 || errors != "" || len(rows) != 6 {
		t.Fatal("multiple record protocol failed")
	}
	if rows[0]["type"] != "ready" || rows[5]["type"] != "summary" || rows[5]["files"] != float64(4) {
		t.Fatal("missing ready/summary")
	}
	for index, row := range rows[1:5] {
		if row["ordinal"] != float64(index+1) {
			t.Fatal("ordinal mismatch")
		}
	}
}

func TestCleanEOFAndNoFindings(t *testing.T) {
	for _, data := range [][]byte{nil, framed([]byte{0, 255, 128, 0}), framed(nil)} {
		exit, rows, errors := protocol(t, data)
		if exit != 0 || errors != "" || rows[len(rows)-1]["findings"] != float64(0) {
			t.Fatal("clean stream failed")
		}
	}
}

func TestIncompleteProtocol(t *testing.T) {
	cases := [][]byte{{1}, {0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 3, 1}, {255, 255, 255, 255, 255, 255, 255, 255}}
	for _, input := range cases {
		exit, rows, errors := protocol(t, input)
		if exit != 2 || errors == "" || len(rows) != 1 {
			t.Fatal("invalid stream was not rejected before result")
		}
		if rows[0]["type"] != "ready" {
			t.Fatal("unexpected protocol output")
		}
	}
}

func TestUnknownArgumentsRejected(t *testing.T) {
	for _, args := range [][]string{nil, {"--other", "value"}, {"--config", "file", "extra"}} {
		var output, errors bytes.Buffer
		if run(args, bytes.NewReader(nil), &output, &errors) != 2 || output.Len() != 0 || !strings.Contains(errors.String(), "invalid_arguments") {
			t.Fatal("invalid arguments accepted")
		}
	}
}

type brokenWriter struct{}

func (brokenWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestOutputFailureIsBlocking(t *testing.T) {
	var errors bytes.Buffer
	if process(bytes.NewReader(nil), brokenWriter{}, &errors, scanner(), fixtureReady) != 2 || !strings.Contains(errors.String(), "output_error") {
		t.Fatal("output failure ignored")
	}
}

func TestInlineSuppressionControl(t *testing.T) {
	d := scanner()
	d.IgnoreGitleaksAllow = false
	fragment := detect.Fragment{Raw: syntheticPAT() + " # gitleaks:allow", FilePath: "record-00000000000000000001"}
	if len(d.Detect(fragment)) != 0 {
		t.Fatal("control did not exercise inline suppression")
	}
	d.IgnoreGitleaksAllow = true
	if len(d.Detect(fragment)) == 0 {
		t.Fatal("inline suppression was honored")
	}
}

func TestReviewedContentRulesRemainActiveWithoutExtensions(t *testing.T) {
	cases := []struct{ id, content string }{
		{"freemius-secret-key", `"secret_key" => "` + "sk_" + "aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC" + `"`},
		{"hashicorp-tf-password", `password = "` + "x7k2p9m" + "4a6t1q8" + `"`},
		{"kubernetes-secret-yaml", "kind: Secret\ndata:\n  fixture: eDdLMnA5TTRhNlQxcTg=\n"},
		{"nuget-config-password", `<add key="Password" value="x7K2p9M4a6T1q8" />`},
	}
	for _, test := range cases {
		t.Run(test.id, func(t *testing.T) {
			exit, rows, errors := protocol(t, framed([]byte(test.content)))
			if exit != 1 || errors != "" || !hasRule(rows, test.id) {
				t.Fatalf("reviewed rule inactive: %s", test.id)
			}
			if fixtureConfig.Rules[test.id].Path != nil {
				t.Fatal("content path prerequisite remains")
			}
		})
	}
	if fixtureConfig.Rules["pkcs12-file"].Path.String() != pkcs12Selector || fixtureConfig.Rules["pkcs12-file"].Regex != nil {
		t.Fatal("path-only policy changed")
	}
	if len(fixtureReady.RemovedContentPathRules) != 4 || fixtureReady.PolicyBeforeSHA256 == fixtureReady.PolicyAfterSHA256 {
		t.Fatal("policy delta not recorded")
	}
}

func TestPolicyDigestIncludesActualPatterns(t *testing.T) {
	serialized, err := json.Marshal(fixtureConfig)
	if err != nil || !bytes.Contains(serialized, []byte(`"Regex":"ghp_[0-9a-zA-Z]{36}"`)) {
		t.Fatal("policy snapshot lost regexp semantics")
	}
}

func TestPolicyDeltaOnlyRemovesReviewedPrerequisites(t *testing.T) {
	original := fixtureConfig
	original.Rules = make(map[string]config.Rule)
	for id, rule := range fixtureConfig.Rules {
		if selector, ok := authorizedContentPaths[id]; ok {
			rule.Path = regexp.MustCompile(selector)
		}
		original.Rules[id] = rule
	}
	before, err := policyDigest(original)
	if err != nil || before != fixtureReady.PolicyBeforeSHA256 {
		t.Fatal("before policy digest mismatch")
	}
	originalRules := make(map[string]config.Rule)
	for id, rule := range original.Rules {
		originalRules[id] = rule
	}
	if _, code := strengthenContentRules(&original); code != "" {
		t.Fatal(code)
	}
	after, err := policyDigest(original)
	if err != nil || after != fixtureReady.PolicyAfterSHA256 {
		t.Fatal("after policy digest mismatch")
	}
	for id, rule := range originalRules {
		if _, approved := authorizedContentPaths[id]; approved {
			rule.Path = nil
		}
		if !reflect.DeepEqual(rule, original.Rules[id]) {
			t.Fatal("policy changed beyond approved Path field")
		}
	}
}

func TestConfigPathCollisionIsBlocking(t *testing.T) {
	d := scanner()
	d.Config.Path = "record-00000000000000000001"
	var output, errors bytes.Buffer
	if process(bytes.NewReader(framed([]byte(syntheticPAT()))), &output, &errors, d, fixtureReady) != 2 || !strings.Contains(errors.String(), "controlled_path_matches_config") {
		t.Fatal("configuration path silently skipped fragment")
	}
}

func TestFingerprintSetCanonicalizationPreservesActualPolicy(t *testing.T) {
	allow := &config.Allowlist{Commits: []string{"second", "first"}, StopWords: []string{"zulu", "alpha"}, RegexTarget: "match", Regexes: []*regexp.Regexp{regexp.MustCompile("first-pattern"), regexp.MustCompile("second-pattern")}}
	rule := config.Rule{RuleID: "synthetic", Regex: regexp.MustCompile("synthetic-pattern"), Entropy: 3, Allowlists: []*config.Allowlist{allow}}
	cfg := config.Config{Rules: map[string]config.Rule{"synthetic": rule}, Allowlists: []*config.Allowlist{allow}}
	original, _ := json.Marshal(cfg)
	first, err := policyDigest(cfg)
	if err != nil {
		t.Fatal(err)
	}
	unchanged, _ := json.Marshal(cfg)
	if !bytes.Equal(original, unchanged) {
		t.Fatal("fingerprinting mutated active policy")
	}
	allow.Commits[0], allow.Commits[1] = allow.Commits[1], allow.Commits[0]
	allow.StopWords[0], allow.StopWords[1] = allow.StopWords[1], allow.StopWords[0]
	second, _ := policyDigest(cfg)
	if first != second {
		t.Fatal("semantic set ordering changed digest")
	}
	allow.Regexes[0], allow.Regexes[1] = allow.Regexes[1], allow.Regexes[0]
	orderedChange, _ := policyDigest(cfg)
	if second == orderedChange {
		t.Fatal("ordered regexp policy was silently canonicalized")
	}
	allow.StopWords = append(allow.StopWords, "new-stopword")
	changed, _ := policyDigest(cfg)
	if changed == orderedChange {
		t.Fatal("changed allowlist semantics did not change digest")
	}
	rule.Entropy += 1
	cfg.Rules["synthetic"] = rule
	entropyChange, _ := policyDigest(cfg)
	if entropyChange == changed {
		t.Fatal("changed entropy did not change digest")
	}
}

func TestReadinessChild(t *testing.T) {
	if os.Getenv("NPA_PRIVATE_SCANNER_READY_CHILD") != "1" {
		return
	}
	data, err := json.Marshal(fixtureReady)
	if err != nil {
		t.Fatal("ready marshal failed")
	}
	os.Stdout.Write(append(data, '\n'))
}

func TestFreshProcessReadyPolicyStability(t *testing.T) {
	var reference []byte
	for index := 0; index < 5; index++ {
		command := exec.Command(os.Args[0], "-test.run=^TestReadinessChild$")
		command.Env = append(os.Environ(), "NPA_PRIVATE_SCANNER_READY_CHILD=1")
		result, err := command.Output()
		if err != nil {
			t.Fatal("fresh readiness child failed")
		}
		line := bytes.SplitN(result, []byte("\n"), 2)[0]
		var decoded ready
		if err := json.Unmarshal(line, &decoded); err != nil {
			t.Fatal("fresh readiness malformed")
		}
		if index == 0 {
			reference = append([]byte(nil), line...)
		} else if !bytes.Equal(reference, line) {
			t.Fatal("fresh-process ready policy changed")
		}
	}
}

func configFixturePath() string {
	if configured := os.Getenv("NPA_IMAGE_BYTE_TEST_CONFIG"); configured != "" {
		return configured
	}
	return filepath.Join("..", "..", "..", "..", ".gitleaks.toml")
}

func TestDescriptorPathExactPolicyAndProtocolParity(t *testing.T) {
	file, err := os.Open(configFixturePath())
	if err != nil {
		t.Fatal("fixture open failed")
	}
	defer file.Close()
	parsed, info, code := configurationFD(int(file.Fd()))
	if code != "" || !reflect.DeepEqual(info, fixtureReady) {
		t.Fatal("descriptor changed exact ready policy")
	}
	if position, err := file.Seek(0, io.SeekCurrent); err != nil || position != 0 {
		t.Fatal("caller descriptor closed or position changed")
	}
	left, _ := policyDigest(parsed)
	right, _ := policyDigest(fixtureConfig)
	if left != right {
		t.Fatal("descriptor changed policy semantics")
	}
	// Upstream config.Translate retains a process-global extension-depth guard.
	// The bridge configures exactly once per process. Compare both real CLI
	// modes in independent children rather than accumulating config loads.
	outputs := make([][]byte, 0, 2)
	for _, mode := range []string{"path", "descriptor"} {
		command := exec.Command(os.Args[0], "-test.run=^TestInheritedDescriptorChild$")
		command.ExtraFiles = []*os.File{file}
		command.Env = append(os.Environ(), "PRIVATE_CONFIG_DESCRIPTOR_CHILD=1",
			"PRIVATE_CONFIG_PAYLOAD=1", "PRIVATE_CONFIG_MODE="+mode)
		body, err := command.Output()
		exitError, ok := err.(*exec.ExitError)
		if !ok || exitError.ExitCode() != 1 || len(exitError.Stderr) != 0 {
			t.Fatal("real config mode did not preserve finding exit status")
		}
		outputs = append(outputs, body)
	}
	if !bytes.Equal(outputs[0], outputs[1]) {
		t.Fatal("path and descriptor full protocol differs")
	}

}

func TestDescriptorModeRejectsAmbiguousOrInvalidArguments(t *testing.T) {
	cases := [][]string{
		{"--config-fd", "-1"}, {"--config-fd", "0"}, {"--config-fd", "1"}, {"--config-fd", "2"},
		{"--config-fd", "invalid"}, {"--config-fd", "18446744073709551616"},
		{"--config", configFixturePath(), "--config-fd", "3"},
		{"--config", "", "--config-fd", "3"}, {"--config-fd", "3", "extra"},
	}
	for _, args := range cases {
		var out, diagnostic bytes.Buffer
		if run(args, bytes.NewReader(nil), &out, &diagnostic) != 2 || out.Len() != 0 ||
			diagnostic.String() != "{\"error\":\"invalid_arguments\"}\n" {
			t.Fatal("ambiguous or invalid descriptor arguments accepted")
		}
	}
}

func TestDescriptorClosedNonregularAndPositionRejected(t *testing.T) {
	for _, fd := range []int{-1, 0, 1, 2} {
		if _, _, code := configurationFD(fd); code != "config_descriptor_invalid" {
			t.Fatal("standard descriptor accepted")
		}
	}
	file, err := os.Open(configFixturePath())
	if err != nil {
		t.Fatal("fixture open failed")
	}
	closed := int(file.Fd())
	file.Close()
	if _, _, code := configurationFD(closed); code != "config_descriptor_invalid" {
		t.Fatal("closed descriptor accepted")
	}
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal("pipe setup failed")
	}
	defer read.Close()
	defer write.Close()
	if _, _, code := configurationFD(int(read.Fd())); code != "config_not_regular" {
		t.Fatal("unseekable pipe accepted")
	}
	directory, err := os.Open(t.TempDir())
	if err != nil {
		t.Fatal("directory setup failed")
	}
	defer directory.Close()
	if _, _, code := configurationFD(int(directory.Fd())); code != "config_not_regular" {
		t.Fatal("directory descriptor accepted")
	}
	positioned, err := os.Open(configFixturePath())
	if err != nil {
		t.Fatal("fixture open failed")
	}
	defer positioned.Close()
	if _, err := positioned.Seek(1, io.SeekStart); err != nil {
		t.Fatal("seek setup failed")
	}
	if _, _, code := configurationFD(int(positioned.Fd())); code != "config_position_invalid" {
		t.Fatal("nonzero descriptor start accepted")
	}
	// Linux O_PATH opens regular inode metadata while denying seek/read access.
	fd, err := syscall.Open(configFixturePath(), 0x200000, 0)
	if err != nil {
		t.Fatal("metadata descriptor setup failed")
	}
	defer syscall.Close(fd)
	if _, _, code := configurationFD(fd); code != "config_position_invalid" {
		t.Fatal("unreadable regular metadata descriptor accepted")
	}
}

func TestStableConfigReadRejectsMutation(t *testing.T) {
	for _, name := range []string{"grow", "shrink", "same_size_restore_mtime", "mode"} {
		t.Run(name, func(t *testing.T) {
			file, err := os.CreateTemp(t.TempDir(), "config-")
			if err != nil {
				t.Fatal("temporary config failed")
			}
			defer file.Close()
			original := []byte("synthetic immutable configuration")
			if _, err := file.WriteAt(original, 0); err != nil {
				t.Fatal("fixture write failed")
			}
			before, _ := file.Stat()
			read := func(reader io.Reader) ([]byte, error) {
				body, err := io.ReadAll(reader)
				if err != nil {
					return nil, err
				}
				switch name {
				case "grow":
					_, err = file.WriteAt([]byte("x"), int64(len(original)))
				case "shrink":
					err = file.Truncate(1)
				case "same_size_restore_mtime":
					// Prove the fixture changed ctime; a short sleep alone does not
					// establish an observable tick on every filesystem.
					changed := false
					for attempt := 0; attempt < 100; attempt++ {
						_, err = file.WriteAt([]byte("X"), 0)
						if err == nil {
							err = os.Chtimes(file.Name(), before.ModTime(), before.ModTime())
						}
						if err != nil {
							break
						}
						current, statErr := file.Stat()
						if statErr != nil {
							t.Fatal("mutation stat failed")
						}
						if current.Sys().(*syscall.Stat_t).Ctim != before.Sys().(*syscall.Stat_t).Ctim {
							changed = true
							break
						}
						time.Sleep(time.Millisecond)
					}
					if err == nil && !changed {
						t.Fatal("fixture did not produce observable ctime change")
					}
				case "mode":
					err = file.Chmod(0400)
				}
				if err != nil {
					t.Fatal("mutation setup failed")
				}
				return body, nil
			}
			if _, code := readConfigData(file, read); code != "config_changed" {
				t.Fatal("config mutation not rejected")
			}
		})
	}
}

func TestConfigReadErrorAndPathGuardsRemain(t *testing.T) {
	file, err := os.Open(configFixturePath())
	if err != nil {
		t.Fatal("fixture open failed")
	}
	defer file.Close()
	if _, code := readConfigData(file, func(io.Reader) ([]byte, error) { return nil, io.ErrUnexpectedEOF }); code != "config_read_error" {
		t.Fatal("config read error accepted")
	}
	directory := t.TempDir()
	link := filepath.Join(directory, "link")
	target, _ := filepath.Abs(configFixturePath())
	if err := os.Symlink(target, link); err != nil {
		t.Fatal("symlink setup failed")
	}
	fifo := filepath.Join(directory, "fifo")
	if err := syscall.Mkfifo(fifo, 0600); err != nil {
		t.Fatal("fifo setup failed")
	}
	for _, path := range []string{link, fifo, directory} {
		if _, _, code := configuration(path); code != "config_not_regular" {
			t.Fatal("existing path guard weakened")
		}
	}
}

func TestInheritedDescriptorChild(t *testing.T) {
	if os.Getenv("PRIVATE_CONFIG_DESCRIPTOR_CHILD") != "1" {
		return
	}
	args := []string{"--config-fd", "3"}
	if os.Getenv("PRIVATE_CONFIG_MODE") == "path" {
		args = []string{"--config", configFixturePath()}
	}
	var input []byte
	if os.Getenv("PRIVATE_CONFIG_PAYLOAD") == "1" {
		input = framed([]byte("public fixture"), []byte(syntheticPAT()))
	}
	var output, diagnostic bytes.Buffer
	exit := run(args, bytes.NewReader(input), &output, &diagnostic)
	os.Stdout.Write(output.Bytes())
	os.Stderr.Write(diagnostic.Bytes())
	os.Exit(exit)
}

func TestActualInheritedDescriptorProtocol(t *testing.T) {
	file, err := os.Open(configFixturePath())
	if err != nil {
		t.Fatal("fixture open failed")
	}
	defer file.Close()
	command := exec.Command(os.Args[0], "-test.run=^TestInheritedDescriptorChild$")
	command.ExtraFiles = []*os.File{file}
	command.Env = append(os.Environ(), "PRIVATE_CONFIG_DESCRIPTOR_CHILD=1")
	body, err := command.Output()
	if err != nil {
		t.Fatal("inherited descriptor child failed")
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	var actual ready
	if decoder.Decode(&actual) != nil || !reflect.DeepEqual(actual, fixtureReady) {
		t.Fatal("child inherited ready policy differs")
	}
	var totals summary
	if decoder.Decode(&totals) != nil || totals.Type != "summary" || totals.Files != 0 {
		t.Fatal("child summary malformed")
	}
	if decoder.Decode(&totals) != io.EOF {
		t.Fatal("unexpected child output")
	}
}
