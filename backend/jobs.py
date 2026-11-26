"""Bounded previews that can be consistently committed as catalog records."""

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic

from .evidence import KIND_TITLES
from .values import DeskError, checked_key, clean_text, key


JOB_CAPACITY = 42
JOB_BYTES = 72 * 1024 * 1024
JOB_LIFETIME = 20 * 60


@dataclass(frozen=True)
class Report:
    kind: str
    settings: dict
    result: dict
    peer: dict | None = None
    images: tuple = ()
    marks_version: str | None = None

    @property
    def size(self):
        return sum(len(file[2]) for file in self.images)


@dataclass
class Job:
    id: str
    scan: dict
    report: Report
    born: float = field(default_factory=monotonic)
    committed: dict | None = None
    access: RLock = field(default_factory=RLock)

    def summary(self):
        return {
            "id": self.id,
            "scan_id": self.scan["id"],
            "kind": self.report.kind,
            "result": self.report.result,
            "settings": self.report.settings,
            "peer_id": self.report.peer["id"] if self.report.peer else None,
            "files": [
                {"index": index, "name": file[0], "type": file[1], "bytes": len(file[2])}
                for index, file in enumerate(self.report.images)
            ],
        }


class Jobs:
    def __init__(self, catalog):
        self.catalog = catalog
        self.entries = OrderedDict()
        self.access = RLock()
        self.used = 0

    def _remove(self, job_id):
        old = self.entries.pop(job_id)
        self.used -= old.report.size

    def _prune(self):
        timestamp = monotonic()
        stale = [identifier for identifier, job in self.entries.items() if timestamp - job.born > JOB_LIFETIME]
        for identifier in stale:
            self._remove(identifier)

    def add(self, scan, report):
        if report.size > JOB_BYTES:
            raise DeskError("The result is too large for a single working preview.", status=413, field="region")
        result = Job(key(), scan, report)
        with self.access:
            self._prune()
            while len(self.entries) >= JOB_CAPACITY or self.used + report.size > JOB_BYTES:
                self._remove(next(iter(self.entries)))
            self.entries[result.id] = result
            self.used += report.size
        return result

    def get(self, job_id):
        checked_key(job_id, "job_id")
        with self.access:
            self._prune()
            job = self.entries.get(job_id)
            if job is None:
                raise DeskError("The working preview expired. Run the operation again.", status=404, code="expired")
            self.entries.move_to_end(job_id)
            return job

    def exhibit(self, job_id, index):
        job = self.get(job_id)
        if not 0 <= index < len(job.report.images):
            raise DeskError("This preview file does not exist.", status=404, code="not_found")
        return job.report.images[index]

    def save(self, job_id, title=None):
        job = self.get(job_id)
        with job.access:
            if job.committed is not None:
                return job.committed, False
            current = job.scan
            expected = {"source_revision": current["revision"], "marks_version": job.report.marks_version}
            if job.report.peer:
                expected["peer_revision"] = job.report.peer["revision"]
            name = clean_text(title, "title", maximum=100) if title is not None else ""
            if not name:
                name = f"{KIND_TITLES[job.report.kind]} · {current['name']}"[:100]
            job.committed = self.catalog.add_record(
                current["id"],
                job.report.kind,
                name,
                job.report.settings,
                job.report.result,
                peer_id=job.report.peer["id"] if job.report.peer else None,
                files=job.report.images,
                expected=expected,
            )
            return job.committed, True
