"""Task-centered local consumer commands. No public execution service."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile


def add_parser(commands: Any) -> None:
    p=commands.add_parser("workload",help="Install and run workload-aware MLX/Metal computation")
    sub=p.add_subparsers(dest="workload_command",required=True)
    sub.add_parser("doctor",help="Execute a tiny Metal computation and report compatibility")
    p=sub.add_parser("demo",help="Run local document search with original example text")
    p.add_argument("--artifact",type=Path)
    p.add_argument("--query",default="GPU precision and verified normalization")
    p=sub.add_parser("search",help="Refresh an embedding index and run top-k search")
    p.add_argument("vectors",type=Path,help="NPZ with documents and queries matrices")
    p.add_argument("--artifact",type=Path)
    p.add_argument("--k",type=int,default=5)
    p=sub.add_parser("export",help="Export a frozen independently confirmed Metal plan")
    p.add_argument("study",type=Path)
    p.add_argument("--output",type=Path,required=True)


def default_artifact() -> Path:
    return Path(__file__).parent/"artifacts/m4-scores-v1"


def application_adapter(artifact: Path | None) -> tuple[Any, str]:
    """Choose an authored operation by manifest data; invalid plans fall back."""
    from .fused_search import FusedPlan
    from .workload import Normalizer
    if artifact is not None:
        path=artifact/'manifest.json' if artifact.is_dir() else artifact
        try:
            if path.stat().st_size<=1_000_000 and json.loads(path.read_text()).get('operation')=='stable_l2_rows_v1':
                return Normalizer(artifact),'row_normalization'
        except (OSError,ValueError,TypeError,AttributeError):
            pass
    return FusedPlan(artifact),'query_scoring'


def doctor() -> dict[str,Any]:
    try:
        import mlx.core as mx
        from .workload import environment_identity,source_identity
        mx.set_default_device(mx.gpu)
        x=mx.arange(128,dtype=mx.float32)
        y=mx.sum(x*x);mx.eval(y);mx.synchronize()
        if y.item()!=690880:
            raise RuntimeError("Metal probe output did not match independent arithmetic")
        return {"metal_executed":True,"environment":environment_identity(),"probe_sum_squares":y.item(),"source_sha256":source_identity(),
                "timing_note":"A correctness probe is not a performance confirmation","bundled_artifact":default_artifact().exists()}
    except (ImportError,RuntimeError) as exc:
        return {"metal_executed":False,"reason":str(exc),"install":"Install kerneldarwin[metal]==0.3.0 on Apple Silicon macOS"}


def run(args: Any) -> dict[str,Any]:
    if args.workload_command=="doctor":return doctor()
    if args.workload_command=="export":
        protocol=json.loads((args.study/'protocol.json').read_text())
        if protocol.get('target')=='fused_query_scores_v1':
            from .fused_export import export_artifact
        else:
            from .workload_export import export_artifact
        return export_artifact(args.study,args.output)
    import mlx.core as mx
    import numpy as np
    from .search import CosineIndex,text_vectors
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(2*1024**3)
    bundled=default_artifact()
    adapter,scope=application_adapter(args.artifact or (bundled if bundled.exists() else None))
    if args.workload_command=="demo":
        texts=[
            "Verify numerical precision using an independent high precision reference.",
            "Metal GPU kernels normalize vectors with stable reductions and readable source.",
            "A cosine search index returns documents matching the query vocabulary.",
            "Unknown shapes and changed environments use a verified MLX library fallback.",
            "Measure full application latency including copies allocation dispatch and results.",
            "A compiler can fuse elementwise operations while leaving reductions separate.",
        ]
        t=time.perf_counter();documents=text_vectors(texts,384);queries=text_vectors([args.query],384)
        mx.eval(documents,queries);encoding_ms=(time.perf_counter()-t)*1000;k=3
    else:
        if args.vectors.stat().st_size>256*1024**2:
            raise ValueError("NPZ exceeds the 256 MiB input limit")
        with zipfile.ZipFile(args.vectors) as archive:
            if sum(x.file_size for x in archive.infolist())>256*1024**2:
                raise ValueError("Uncompressed NPZ exceeds the 256 MiB input limit")
        t=time.perf_counter()
        with np.load(args.vectors,allow_pickle=False) as file:
            docs,qs=file["documents"],file["queries"]
            if docs.dtype not in (np.float32,np.float16) or qs.dtype!=docs.dtype:
                raise TypeError("Both matrices must share explicit float32 or float16 dtype")
            documents,queries=mx.array(docs),mx.array(qs)
        encoding_ms=(time.perf_counter()-t)*1000;k=args.k;texts=None
    mx.synchronize();start=time.perf_counter()
    index=adapter.index(documents) if scope=='query_scoring' else CosineIndex(documents,adapter)
    ids,scores=index.search(queries,k)
    mx.synchronize();compute_ms=(time.perf_counter()-start)*1000
    result={"application":"cosine index refresh + top-k search","artifact_status":adapter.status,"dispatch_scope":scope,
        "input_prepare_ms":encoding_ms,"compute_ms":compute_ms,"calls":adapter.calls,"fallback_calls":adapter.fallback_calls,
        "last_decision":asdict(adapter.last_decision),"ids":ids,"scores":scores,
        "timing_note":"Single invocation including cold work; not an acceleration claim"}
    if texts:result["matches"]=[texts[i] for i in ids[0]]
    return result
