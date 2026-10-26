"""Disk-backed audit values and bounded-memory JSON serialization.

Archive files are ordinary JSON with references to previously archived values.
A small byte-offset index lets the serializer expand those references without
parsing the containing file or retaining completed episode trees in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid


_CHUNK = 64 * 1024


def _tokens(value, *, canonical, indent, normalize_nonfinite=True, level=0, active=None):
    """Yield JSON text or an unexpanded archived value."""
    if isinstance(value, ArchivedRecord):
        yield value
        return
    if active is None:
        active = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            if not normalize_nonfinite:
                raise ValueError("Out of range float values are not JSON compliant")
            value = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        yield json.dumps(value, ensure_ascii=True, allow_nan=False)
        return
    identity = id(value)
    if identity in active:
        raise ValueError("Circular reference in JSON audit value")
    active.add(identity)
    try:
        if isinstance(value, ArchivedSequence):
            items = value.references()
            mapping = False
        elif isinstance(value, Mapping):
            keys = sorted(value) if canonical else value
            items = ((key, value[key]) for key in keys)
            mapping = True
        elif isinstance(value, (tuple, list)):
            items, mapping = iter(value), False
        elif hasattr(value, "as_record"):
            yield from _tokens(value.as_record(), canonical=canonical, indent=indent,
                               normalize_nonfinite=normalize_nonfinite, level=level, active=active)
            return
        elif is_dataclass(value):
            yield from _tokens({field.name: getattr(value, field.name) for field in fields(value)},
                               canonical=canonical, indent=indent, normalize_nonfinite=normalize_nonfinite,
                               level=level, active=active)
            return
        else:
            raise TypeError(f"{type(value).__name__} is not JSON serializable")
        yield "{" if mapping else "["
        first = True
        for item in items:
            if not first:
                yield ","
            if indent is not None:
                yield "\n" + " " * (indent * (level + 1))
            first = False
            if mapping:
                key, item = item
                # Match the JSON encoder's permitted key conversions.
                if not isinstance(key, str):
                    if key is None:
                        key = "null"
                    elif isinstance(key, bool):
                        key = "true" if key else "false"
                    elif isinstance(key, (int, float)):
                        key = json.dumps(key, allow_nan=False)
                    else:
                        raise TypeError("JSON object keys must be strings or scalar numbers")
                yield json.dumps(key, ensure_ascii=True)
                yield ": " if indent is not None else ":"
            yield from _tokens(item, canonical=canonical, indent=indent,
                               normalize_nonfinite=normalize_nonfinite, level=level + 1, active=active)
        if not first and indent is not None:
            yield "\n" + " " * (indent * level)
        yield "}" if mapping else "]"
    finally:
        active.remove(identity)


def iter_json(value, *, canonical=False, indent=None, normalize_nonfinite=True):
    """Stream strict JSON; archived subtrees expand to their original values.

    Archive blocks are stored in canonical compact form. Pretty output keeps
    these blocks compact, without changing the expanded JSON data or hash.
    Non-finite diagnostics use the existing CLI's explicit string spelling.
    """
    for token in _tokens(value, canonical=canonical, indent=indent,
                         normalize_nonfinite=normalize_nonfinite):
        if isinstance(token, ArchivedRecord):
            yield from token.iter_json()
        else:
            yield token


def write_json(stream, value, *, canonical=False, indent=None, normalize_nonfinite=True):
    for chunk in iter_json(value, canonical=canonical, indent=indent,
                           normalize_nonfinite=normalize_nonfinite):
        stream.write(chunk)


@dataclass(frozen=True)
class ArchivedRecord:
    path: Path
    sha256: str
    index_sha256: str

    def as_record(self):
        return self

    def descriptor(self):
        return {"file": self.path.name, "sha256": self.sha256,
                "index_sha256": self.index_sha256}

    def iter_json(self):
        index_bytes = self.path.with_suffix(".index.json").read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != self.index_sha256:
            raise ValueError(f"Archive index integrity failure: {self.path}")
        references = json.loads(index_bytes)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for entry in references:
                remaining = entry["offset"] - stream.tell()
                if remaining < 0:
                    raise ValueError(f"Invalid archive offsets: {self.path}")
                while remaining:
                    chunk = stream.read(min(_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"Truncated archive: {self.path}")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    yield chunk.decode("ascii")
                marker = stream.read(entry["length"])
                digest.update(marker)
                reference = entry["reference"]
                name = reference["file"]
                if Path(name).name != name:
                    raise ValueError("Archive reference must stay in its archive directory")
                child = ArchivedRecord(self.path.parent / name, reference["sha256"],
                                       reference["index_sha256"])
                yield from child.iter_json()
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
                yield chunk.decode("ascii")
        if digest.hexdigest() != self.sha256:
            raise ValueError(f"Archive integrity failure: {self.path}")

    def load(self):
        """Explicitly materialize one record, for record-by-record inspection."""
        return json.loads("".join(self.iter_json()))


class ArchivedSequence(Sequence):
    """Append completed records; ordinary iteration loads only one at a time."""

    def __init__(self, archive, name, initial=()):
        self.archive = archive
        self.name = str(name)
        self._references = []
        for value in initial:
            self.append(value)

    def append(self, value):
        self._references.append(self.archive.store(value))

    def __len__(self):
        return len(self._references)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [reference.load() for reference in self._references[index]]
        return self._references[index].load()

    def __iter__(self):
        for reference in self._references:
            yield reference.load()

    def references(self):
        return iter(self._references)

    def as_record(self):
        return self


class JsonArchive:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)

    def sequence(self, name, initial=()):
        return ArchivedSequence(self, name, initial)

    def store(self, value):
        if isinstance(value, ArchivedRecord):
            return value
        name = uuid.uuid4().hex
        path = self.path / f"{name}.json"
        index_path = path.with_suffix(".index.json")
        references = []
        digest = hashlib.sha256()
        temporary = None
        index_temporary = None
        try:
            fd, temporary = tempfile.mkstemp(prefix=".write-", dir=self.path)
            with os.fdopen(fd, "wb") as stream:
                for token in _tokens(value, canonical=True, indent=None):
                    if isinstance(token, ArchivedRecord):
                        if token.path.parent != self.path:
                            raise ValueError("Nested archive values must share an archive directory")
                        reference = token.descriptor()
                        data = json.dumps({"$archive": reference}, sort_keys=True,
                                          ensure_ascii=True, separators=(",", ":")).encode("ascii")
                        references.append({"offset": stream.tell(), "length": len(data),
                                           "reference": reference})
                    else:
                        data = token.encode("ascii")
                    stream.write(data)
                    digest.update(data)
                stream.flush()
                os.fsync(stream.fileno())
            index_bytes = json.dumps(references, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode("ascii")
            fd, index_temporary = tempfile.mkstemp(prefix=".write-", dir=self.path)
            with os.fdopen(fd, "wb") as stream:
                stream.write(index_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.replace(index_temporary, index_path)
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return ArchivedRecord(path, digest.hexdigest(), hashlib.sha256(index_bytes).hexdigest())
        finally:
            for unfinished in (temporary, index_temporary):
                if unfinished is not None:
                    Path(unfinished).unlink(missing_ok=True)
