#!/usr/bin/env python3
"""Check that a wheel ships the exact authored workload runtime and source pins."""
from __future__ import annotations
import argparse
from email.parser import Parser
from pathlib import Path
import zipfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'src/kerneldarwin'
    checked=[]
    with zipfile.ZipFile(args.wheel) as wheel:
        paths=[*source.glob('*.py'),*source.glob('*.json'),*source.glob('metal/*.metal'),source/'py.typed']
        for path in paths:
            name='kerneldarwin/'+str(path.relative_to(source))
            if wheel.read(name)!=path.read_bytes():raise AssertionError('Wheel runtime/source mismatch: '+name)
            checked.append(name)
        for name in ('workload.html','workload.css','workload.js','data/workload.json','data/workload-demo.json'):
            if wheel.read('kerneldarwin/lab/'+name)!=(root/'lab'/name).read_bytes():
                raise AssertionError('Wheel interface mismatch: '+name)
            checked.append(name)
        metadata_names=[n for n in wheel.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata_names)!=1:raise AssertionError('Ambiguous wheel metadata')
        metadata=Parser().parsestr(wheel.read(metadata_names[0]).decode())
        dependencies=metadata.get_all('Requires-Dist',[])
        for requirement in ('mlx==0.32.0','mlx-metal==0.32.0','numpy==2.3.5'):
            if not any(line.startswith(requirement) and 'extra == "metal"' in line for line in dependencies):
                raise AssertionError('Missing pinned Metal dependency: '+requirement)
    print(f'PASS: {len(checked)} exact workload runtime/pin/Metal/interface files and all three Metal dependency pins')

if __name__=='__main__':main()
