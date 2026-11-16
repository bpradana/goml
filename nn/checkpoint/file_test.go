package checkpoint

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteFile_AtomicReplacementCreatesParentsAndPrivateFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "model.goml")
	for _, contents := range []string{"first", "replacement"} {
		if err := writeFile(path, []byte(contents)); err != nil {
			t.Fatal(err)
		}

		data, err := os.ReadFile(path)
		if err != nil || string(data) != contents {
			t.Fatalf("got %q: %v", data, err)
		}
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}

	if info.Mode().Perm() != 0o600 {
		t.Fatalf("permissions %o, want 600", info.Mode().Perm())
	}

	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil || len(entries) != 1 {
		t.Fatalf("temporary files remain: %v %v", entries, err)
	}
}

func TestWriteFile_RejectsBadDestinationsAndCleansTemporaryFiles(t *testing.T) {
	if err := writeFile("", []byte("data")); err == nil {
		t.Fatal("accepted empty path")
	}

	directory := t.TempDir()

	destination := filepath.Join(directory, "existing-directory")
	if err := os.Mkdir(destination, 0o755); err != nil {
		t.Fatal(err)
	}

	if err := writeFile(destination, []byte("data")); err == nil {
		t.Fatal("replaced directory with checkpoint")
	}

	entries, err := os.ReadDir(directory)
	if err != nil || len(entries) != 1 {
		t.Fatalf("failed write left temporary files: %v %v", entries, err)
	}

	file := filepath.Join(directory, "regular-file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	if err := writeFile(filepath.Join(file, "model.goml"), nil); err == nil {
		t.Fatal("accepted a file as parent directory")
	}

	if _, err := readFile(filepath.Join(directory, "absent")); err == nil {
		t.Fatal("accepted missing checkpoint")
	}
}
