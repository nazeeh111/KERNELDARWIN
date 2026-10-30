#!/usr/bin/env python3
"""Refresh the installed measurement-source pin after an intentional source edit.

CPU CI checks equality; a wheel can verify source-bound studies without having
the development checkout. This metadata never supplies code for execution.
"""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for script,metadata in (('workload_study.py','study-source.json'),('fused_study.py','fused-study-source.json')):
    pin={'measurement_script_sha256':hashlib.sha256((root/'scripts'/script).read_bytes()).hexdigest()}
    (root/'src/kerneldarwin'/metadata).write_text(json.dumps(pin,indent=2)+'\n')
    print(json.dumps({'metadata':metadata,**pin}))
