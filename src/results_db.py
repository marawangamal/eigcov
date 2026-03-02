"""Append-only results database backed by a JSON lines file."""

import hashlib
import json
import os
from datetime import datetime


def args_to_dict(args):
    """Convert an argparse Namespace to a JSON-serializable dict."""
    return {
        k: v if isinstance(v, (bool, int, float, str, list, type(None))) else str(v)
        for k, v in vars(args).items()
    }


def make_run_hash(script, args, ignore=None):
    """Stable hash of (script, args) used to identify a unique run.

    Args:
        ignore: optional set of arg keys to exclude from the hash (e.g.
                training-only params like lr/wd, environment paths, or fields
                that are set dynamically after the hash is computed).
    """
    d = args_to_dict(args)
    if ignore:
        d = {k: v for k, v in d.items() if k not in ignore}
    payload = json.dumps({"script": script, **d}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record_exists(db_path, run_hash):
    """Return True if any record with the given run_hash exists in the DB."""
    if not os.path.exists(db_path):
        return False
    with open(db_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if json.loads(line).get("run_hash") == run_hash:
                return True
    return False


def append_result(db_path, record, run_hash):
    """Append one result record to a JSON lines file.

    Each line is a self-contained JSON object.  Load the whole file with:
        import pandas as pd
        df = pd.read_json(db_path, lines=True)
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    record = {"run_hash": run_hash, "timestamp": datetime.now().isoformat(), **record}
    with open(db_path, "a") as f:
        f.write(json.dumps(record) + "\n")
