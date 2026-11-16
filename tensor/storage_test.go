package tensor

import (
	"sync/atomic"
	"testing"
	"time"
)

type releaseBarrier struct {
	started chan struct{}
	finish  chan struct{}
	calls   atomic.Int32
}

func (*releaseBarrier) Len() int {
	return 1
}

func (*releaseBarrier) DeviceID() string {
	return "test"
}

func (*releaseBarrier) Valid() error {
	return nil
}

func (b *releaseBarrier) Close() error {
	b.calls.Add(1)
	close(b.started)
	<-b.finish

	return nil
}

func TestStorage_CloseWaitsForConcurrentCleanupReferenceRelease(t *testing.T) {
	buffer := &releaseBarrier{started: make(chan struct{}), finish: make(chan struct{})}
	storage := &deviceStorage{buffer: buffer}
	storage.refs.Store(1)
	lease := &storageLease{storage: storage}
	cleanupDone := make(chan struct{})

	go func() {
		_ = lease.close()
		close(cleanupDone)
	}()

	<-buffer.started

	explicitDone := make(chan struct{})

	go func() {
		_ = lease.close()
		close(explicitDone)
	}()

	select {
	case <-explicitDone:
		close(buffer.finish)
		t.Fatal("Close returned while the last reference was still being released")
	case <-time.After(10 * time.Millisecond):
	}

	close(buffer.finish)
	<-cleanupDone
	<-explicitDone

	if buffer.calls.Load() != 1 || storage.refs.Load() != 0 {
		t.Fatal("allocation was not released exactly once")
	}
}
