"""Read-only interrupted execution reconstruction, adapted from OpenChia.

The journal adapter supplies admitted frames in their original global order.
Matching saved replies does not authorize live work or replay side effects.
"""

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Mapping

from .identities import EpisodeRef, normalize_structural_path


_BOUNDARIES = {"episode_started", "unit_completed", "episode_completed"}


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _copy(value):
    return json.loads(_canonical(value))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class ReconstructionError(RuntimeError):
    """The saved execution cannot be reconstructed without changing meaning."""


class ReconstructionBoundaryReached(ReconstructionError):
    """Saved frames matched; the host must separately authorize continuation."""


@dataclass(frozen=True)
class RecordedReply:
    frame_type: str
    body: object
    event_ref: object


@dataclass(frozen=True)
class ExpectedFrame:
    frame_type: str
    body: object
    event_ref: object
    reply: RecordedReply | None = None


def _path(raw):
    if isinstance(raw, (tuple, list)) and raw and all(isinstance(part, Mapping) for part in raw):
        if any(set(part) != {"grain", "key"} for part in raw):
            raise ReconstructionError("episode path has unexpected fields")
        raw = [(part["grain"], part["key"]) for part in raw]
    try:
        return normalize_structural_path(raw)
    except (TypeError, ValueError) as exc:
        raise ReconstructionError("invalid reconstruction episode path") from exc


def _active_stack(frames, logical_run_id, *, allow_preflight):
    stack, seen, returning = [], set(), None
    pending = None
    for index, frame in enumerate(frames):
        body = frame.body
        if not isinstance(body, Mapping):
            raise ReconstructionError("reconstruction frame body must be a mapping")
        if frame.frame_type == "trace_event":
            if frame.reply is not None:
                raise ReconstructionError("a diagnostic trace frame cannot have a reply")
            continue
        if pending is not None:
            raise ReconstructionError("an unanswered request must be the final worker frame")
        if frame.frame_type == "run_event":
            if frame.reply is not None or set(body) != {"episode_id", "event_kind", "payload"}:
                raise ReconstructionError("invalid reconstructed Episode event envelope")
            kind, payload = body["event_kind"], body["payload"]
            if kind not in _BOUNDARIES or not isinstance(payload, Mapping):
                raise ReconstructionError("unsupported reconstructed Episode boundary")
            if kind == "episode_started":
                path = _path(payload.get("episode_path"))
                episode_id = EpisodeRef(logical_run_id, path).episode_id
                if (episode_id != body["episode_id"] or episode_id in seen
                        or (stack and path[:-1] != stack[-1]["episode_path"])
                        or (not stack and (seen or len(path) != 1))):
                    raise ReconstructionError("recording does not contain one serial nested Episode stack")
                required = {"request", "goal_view", "goal_state_id", "initial_goal_state_id"}
                if not required <= set(payload):
                    raise ReconstructionError("Episode start lacks its request or goal-state boundary")
                seen.add(episode_id)
                stack.append({"episode_id": episode_id, "episode_path": path})
                returning = None
            else:
                if not stack or stack[-1]["episode_id"] != body["episode_id"]:
                    raise ReconstructionError("worker boundary differs from the active nested Episode")
                returning = stack.pop() if kind == "episode_completed" else None
            continue
        if not frame.frame_type.endswith("_request"):
            raise ReconstructionError("unsupported reconstruction frame type")
        if (not {"episode_id", "episode_path", "request_id", "request"} <= set(body)
                or not isinstance(body["request_id"], str) or not body["request_id"]
                or not isinstance(body["request"], Mapping)):
            raise ReconstructionError("invalid reconstructed request envelope")
        if body.get("preflight") is True:
            if (not allow_preflight or seen or stack or body.get("episode_id") is not None
                    or body.get("episode_path") != []):
                raise ReconstructionError("preflight request must precede every Episode")
        else:
            path = _path(body.get("episode_path"))
            if EpisodeRef(logical_run_id, path).episode_id != body.get("episode_id"):
                raise ReconstructionError("request path differs from its logical Episode identity")
            active = stack and stack[-1]["episode_id"] == body["episode_id"]
            result = returning and returning["episode_id"] == body["episode_id"]
            if not (active or result):
                raise ReconstructionError("request is not from the active or just-completed Episode")
            if active:
                returning = None
        if frame.reply is None:
            pending = index
        elif (frame.reply.frame_type != frame.frame_type.removesuffix("_request") + "_response"
                or not isinstance(frame.reply.body, Mapping)
                or frame.reply.body.get("request_id") != body["request_id"]):
            raise ReconstructionError("recorded response does not match its request")
    return stack, pending


class ReconstructionCursor:
    """Verify one exact ordered prefix; never fetch, mutate, or continue it."""

    def __init__(self, frames, logical_run_id, *, allow_preflight=False):
        if not isinstance(logical_run_id, str) or not logical_run_id:
            raise ReconstructionError("logical_run_id must be nonempty text")
        saved = []
        try:
            for frame in frames:
                if not isinstance(frame.frame_type, str) or not frame.frame_type:
                    raise ReconstructionError("frame_type must be nonempty text")
                reply = frame.reply
                saved.append(ExpectedFrame(frame.frame_type, _copy(frame.body), _copy(frame.event_ref),
                    None if reply is None else RecordedReply(reply.frame_type,
                        _copy(reply.body), _copy(reply.event_ref))))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ReconstructionError("invalid admitted reconstruction frame") from exc
        self._frames = tuple(saved)
        self._stack, self._pending_index = _active_stack(
            self._frames, logical_run_id, allow_preflight=allow_preflight)
        self._position = 0
        self._divergence = None

    @property
    def prefix_verified(self):
        return self._divergence is None and self._position == len(self._frames)

    @property
    def retry_pending_request(self):
        return self.prefix_verified and self._pending_index is not None

    def accept(self, frame):
        if self._divergence is not None:
            raise ReconstructionError(_canonical(self._divergence))
        if self.prefix_verified:
            raise ReconstructionBoundaryReached("saved prefix matched; live work has not been authorized")
        expected = self._frames[self._position]
        try:
            matched = frame.frame_type == expected.frame_type and _canonical(frame.body) == _canonical(expected.body)
            observed_hash = _digest(frame.body)
            observed_type = frame.frame_type
        except (AttributeError, TypeError, ValueError) as exc:
            self._divergence = {"kind": "invalid_reconstruction_frame",
                                "expected_event_ref": expected.event_ref}
            raise ReconstructionError(_canonical(self._divergence)) from exc
        if not matched:
            self._divergence = {
                "kind": "reconstruction_diverged", "expected_event_ref": expected.event_ref,
                "expected_frame_type": expected.frame_type, "observed_frame_type": observed_type,
                "expected_body_hash": _digest(expected.body), "observed_body_hash": observed_hash}
            raise ReconstructionError(_canonical(self._divergence))
        self._position += 1
        reply = expected.reply
        return None if reply is None else RecordedReply(reply.frame_type, _copy(reply.body), _copy(reply.event_ref))

    def report(self):
        pending = None if self._pending_index is None else self._frames[self._pending_index]
        return _copy({
            "matched_worker_frames": self._position,
            "remaining_worker_frames": len(self._frames) - self._position,
            "expected_active_stack": self._stack,
            "status": "diverged" if self._divergence else (
                "prefix_verified" if self.prefix_verified else "reconstructing"),
            "divergence": self._divergence,
            "live_work_authorized": False,
            "pending_request_ref": None if pending is None else pending.event_ref,
        })
