# metabus.py
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
from threading import RLock
import copy

@dataclass
class _State:
    iteration: int = -1
    version: int = 0
    data: Dict[Tuple[str, str], Any] = field(default_factory=dict)      # (ns, key) -> value
    tracks: Dict[str, Dict[str, Any]] = field(default_factory=dict)     # track_id -> {key: value}

class MetaBus:
    def __init__(self) -> None:
        self._lock = RLock()
        self._state = _State()

    # lifecycle
    def begin(self, iteration: int) -> None:
        with self._lock:
            self._state.iteration = iteration
            self._state.version += 1
            self._state.data.clear()
            self._state.tracks.clear()          # reset per real video frame

    # global (non-track) metadata
    def put(self, key: str, value: Any, ns: str = "default") -> None:
        with self._lock:
            self._state.data[(ns, key)] = value

    def get(self, key: str, default=None, ns: str = "default") -> Any:
        with self._lock:
            return self._state.data.get((ns, key), default)

    # track-scoped metadata
    def put_track(self, track_id: int | str, key: str, value: Any) -> None:
        tid = str(track_id)
        with self._lock:
            self._state.tracks.setdefault(tid, {})[key] = value

    def get_track(self, track_id: int | str, key: str, default=None) -> Any:
        tid = str(track_id)
        with self._lock:
            return self._state.tracks.get(tid, {}).get(key, default)

    def iter_tracks(self):
        with self._lock:
            # shallow copy of keys to allow safe iteration
            return list(self._state.tracks.keys())

    # safe snapshot for recorder/inspectors
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            # deep-copy small dicts to decouple from mutations
            return {
                "iteration": self._state.iteration,
                "data": copy.deepcopy(self._state.data),
                "tracks": copy.deepcopy(self._state.tracks),
                "version": self._state.version,
            }

    @property
    def iteration(self) -> int:
        return self._state.iteration

    @property
    def version(self) -> int:
        return self._state.version

bus = MetaBus()
