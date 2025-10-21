# metabus_record.py
from typing import Any, Dict, List, Tuple
import json, datetime as dt

def _jsonify(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.generic): return obj.item()
        if isinstance(obj, np.ndarray): return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, (bytes, bytearray)): return obj.decode("utf-8", "replace")
    if isinstance(obj, (set, tuple)): return list(obj)
    if isinstance(obj, (dt.datetime, dt.date)): return obj.isoformat()
    if isinstance(obj, dict): return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_jsonify(x) for x in obj]
    return obj

def _denorm(ns_key_map: Dict[Tuple[str, str], Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for (ns, key), val in ns_key_map.items():
        out.setdefault(ns, {})[key] = val
    return out

class MetaBusRecorder:
    def __init__(self, keep_in_memory: bool = True, jsonl_path: str | None = None):
        self.keep_in_memory = keep_in_memory
        self.jsonl_path = jsonl_path
        self.frames: List[Dict[str, Any]] = []
        if self.jsonl_path:
            open(self.jsonl_path, "w").close()

    def commit(self, bus) -> None:
        snap = bus.snapshot()
        # tidy for output
        tidy = {
            "Frame": snap["iteration"],
            "globals": _denorm({_k: _jsonify(v) for _k, v in snap["data"].items()}),
            "tracks": {tid: {k: _jsonify(v) for k, v in td.items()} for tid, td in snap["tracks"].items()},
        }
        if self.keep_in_memory:
            self.frames.append(tidy)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(tidy, ensure_ascii=False) + "\n")

    def save_json(self, path: str, pretty: bool = False) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.frames, f, indent=2 if pretty else None, ensure_ascii=False)
