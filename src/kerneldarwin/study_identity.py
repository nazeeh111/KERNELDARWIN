"""Bind performance decisions to the application, alternatives and study code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
_LOADED_BINDING_IDENTITY=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def implementation_identity() -> str:
    from .workload import source_identity
    from .search import _LOADED_APPLICATION_IDENTITY
    from .library_baselines import _LOADED_BASELINE_IDENTITY
    pin=json.loads((Path(__file__).parent/'study-source.json').read_text())
    script=pin['measurement_script_sha256']
    if not isinstance(script,str) or len(script)!=64 or any(c not in '0123456789abcdef' for c in script):
        raise ValueError('Invalid installed study implementation identity')
    return hashlib.sha256(json.dumps({'operation':source_identity(),'application':_LOADED_APPLICATION_IDENTITY,
        'baselines':_LOADED_BASELINE_IDENTITY,'measurement_script':script,'binding':_LOADED_BINDING_IDENTITY},sort_keys=True).encode()).hexdigest()
