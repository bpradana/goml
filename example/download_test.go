package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/md5"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestDownloadVerifiesChecksumAndExtractsOnlyDatasetFiles(t *testing.T) {
	var archive bytes.Buffer

	gzipWriter := gzip.NewWriter(&archive)
	tarWriter := tar.NewWriter(gzipWriter)

	for _, name := range batchNames {
		contents := fixtureRecords(1)
		if err := tarWriter.WriteHeader(&tar.Header{Name: "cifar-10-batches-bin/" + name, Mode: 0o644, Size: int64(len(contents))}); err != nil {
			t.Fatal(err)
		}

		if _, err := tarWriter.Write(contents); err != nil {
			t.Fatal(err)
		}
	}

	if err := tarWriter.Close(); err != nil {
		t.Fatal(err)
	}

	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(archive.Bytes()) }))
	defer server.Close()

	destination := filepath.Join(t.TempDir(), "dataset")
	if err := downloadDataset(t.Context(), destination, server.URL, "wrong-checksum"); err == nil {
		t.Fatal("accepted checksum mismatch")
	}

	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatal("published unverified data", err)
	}

	checksum := fmt.Sprintf("%x", md5.Sum(archive.Bytes()))
	if err := downloadDataset(t.Context(), destination, server.URL, checksum); err != nil {
		t.Fatal(err)
	}

	contents, err := os.ReadFile(filepath.Join(destination, "test_batch.bin"))
	if err != nil || !bytes.Equal(contents, fixtureRecords(1)) {
		t.Fatal("wrong extracted data", err)
	}
}

func TestExtractionRejectsTraversalAndLinks(t *testing.T) {
	for _, header := range []*tar.Header{
		{Name: "../escaped", Mode: 0o644, Size: 0},
		{Name: "cifar-10-batches-bin/test_batch.bin", Typeflag: tar.TypeSymlink, Linkname: "../escaped"},
	} {
		var archive bytes.Buffer

		gz := gzip.NewWriter(&archive)

		tw := tar.NewWriter(gz)
		if err := tw.WriteHeader(header); err != nil {
			t.Fatal(err)
		}

		if err := tw.Close(); err != nil {
			t.Fatal(err)
		}

		if err := gz.Close(); err != nil {
			t.Fatal(err)
		}

		if err := extractDataset(bytes.NewReader(archive.Bytes()), t.TempDir()); err == nil {
			t.Fatal("accepted unsafe archive entry", header.Name)
		}
	}
}
