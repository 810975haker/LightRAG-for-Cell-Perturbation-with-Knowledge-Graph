from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import time
import gzip


DEFAULT_CATALOG = {
    "datasets": [
        {
            "name": "tcga_luad_expression",
            "group": "tcga_luad",
            "modality": "expression",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUAD.htseq_counts.tsv.gz",
                "TCGA-LUAD.htseq_fpkm.tsv.gz",
                "TCGA-LUAD.htseq_fpkm-uq.tsv.gz",
            ],
            "hub_urls": [
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_luad_mirna",
            "group": "tcga_luad",
            "modality": "mirna",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUAD.mirna.tsv.gz",
                "TCGA-LUAD.miRNA.tsv.gz",
                "TCGA-LUAD.miRNA_gene_expression.tsv.gz",
            ],
            "hub_urls": [
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_luad_cnv",
            "group": "tcga_luad",
            "modality": "cnv",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUAD.GISTIC2_CopyNumber_Gistic2_all_thresholded.by_genes.tsv.gz",
                "TCGA-LUAD.GISTIC2_CopyNumber_Gistic2_all_thresholded.by_genes.tsv",
            ],
            "hub_urls": [
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_luad_methylation",
            "group": "tcga_luad",
            "modality": "methylation",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUAD.methylation450.tsv.gz",
                "TCGA-LUAD.methylation450.tsv",
            ],
            "hub_urls": [
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_luad_rppa",
            "group": "tcga_luad",
            "modality": "proteomics",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUAD.RPPA.tsv.gz",
                "TCGA-LUAD.RPPA.tsv",
            ],
            "hub_urls": [
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_lusc_expression",
            "group": "tcga_lusc",
            "modality": "expression",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUSC.htseq_counts.tsv.gz",
                "TCGA-LUSC.htseq_fpkm.tsv.gz",
                "TCGA-LUSC.htseq_fpkm-uq.tsv.gz",
            ],
            "hub_urls": [
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_lusc_mirna",
            "group": "tcga_lusc",
            "modality": "mirna",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUSC.mirna.tsv.gz",
                "TCGA-LUSC.miRNA.tsv.gz",
                "TCGA-LUSC.miRNA_gene_expression.tsv.gz",
            ],
            "hub_urls": [
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_lusc_cnv",
            "group": "tcga_lusc",
            "modality": "cnv",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUSC.GISTIC2_CopyNumber_Gistic2_all_thresholded.by_genes.tsv.gz",
                "TCGA-LUSC.GISTIC2_CopyNumber_Gistic2_all_thresholded.by_genes.tsv",
            ],
            "hub_urls": [
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_lusc_methylation",
            "group": "tcga_lusc",
            "modality": "methylation",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUSC.methylation450.tsv.gz",
                "TCGA-LUSC.methylation450.tsv",
            ],
            "hub_urls": [
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "tcga_lusc_rppa",
            "group": "tcga_lusc",
            "modality": "proteomics",
            "matrix_format": "tsv",
            "file_candidates": [
                "TCGA-LUSC.RPPA.tsv.gz",
                "TCGA-LUSC.RPPA.tsv",
            ],
            "hub_urls": [
                "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/",
                "https://gdc-hub.s3.us-east-1.amazonaws.com/download/",
            ],
            "source": "TCGA",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "ccle_expression",
            "group": "ccle",
            "modality": "expression",
            "matrix_format": "tsv",
            "file_candidates": [
                "CCLE_expression_full.tsv.gz",
                "CCLE_expression_full.tsv",
            ],
            "hub_urls": [
                "https://ucscpublic.xenahubs.net/download/",
            ],
            "source": "CCLE",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "ccle_cnv",
            "group": "ccle",
            "modality": "cnv",
            "matrix_format": "tsv",
            "file_candidates": [
                "CCLE_CNV_2013-12-03.gistic.tsv.gz",
                "CCLE_CNV_2013-12-03.gistic.tsv",
            ],
            "hub_urls": [
                "https://ucscpublic.xenahubs.net/download/",
            ],
            "source": "CCLE",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "ccle_rppa",
            "group": "ccle",
            "modality": "proteomics",
            "matrix_format": "tsv",
            "file_candidates": [
                "CCLE_RPPA_20180123.tsv.gz",
                "CCLE_RPPA_20180123.tsv",
            ],
            "hub_urls": [
                "https://ucscpublic.xenahubs.net/download/",
            ],
            "source": "CCLE",
            "version": "current",
            "min_value": 0.0,
        },
        {
            "name": "cptac_luad_proteomics",
            "group": "cptac_luad",
            "modality": "proteomics",
            "matrix_format": "tsv",
            "file_candidates": [
                "CPTAC-LUAD-proteomics.tsv.gz",
                "CPTAC-LUAD-proteomics.tsv",
            ],
            "hub_urls": [
                "https://cptac.xenahubs.net/download/",
            ],
            "source": "CPTAC",
            "version": "current",
            "min_value": 0.0,
        },
    ],
    "target_maps": [
        {
            "name": "mirdb_mirna_targets",
            "url": "https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz",
            "filename": "miRDB_v6.0_prediction_result.txt.gz",
            "source": "miRDB",
            "version": "v6.0",
            "license": "miRDB terms",
            "delimiter": "\t"
        }
    ]
}


@dataclass
class DownloadResult:
    name: str
    status: str
    url: str
    path: str
    error: str = ""
    checksum_sha256: str = ""
    retrieved_at_utc: str = ""


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_file(url: str, output_path: Path, timeout: int = 180, attempts: int = 2,
                  chunk_size: int = 1024 * 1024) -> Tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, timeout), headers=headers) as resp:
                # figshare API 可能返回 202（异步准备），不视为失败
                if resp.status_code == 202:
                    time.sleep(3)
                    continue
                resp.raise_for_status()
                total = 0
                with output_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                if total == 0:
                    raise RuntimeError("Downloaded 0 bytes (empty response body)")
            return True, ""
        except Exception as exc:
            last_error = str(exc)
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            if attempt < attempts:
                time.sleep(2.0 * attempt)
                continue
            return False, last_error
    return False, last_error


def download_figshare_file(file_id: str, output_path: Path, timeout: int = 300) -> Tuple[bool, str]:
    """通过 figshare API 下载文件（处理 202 异步准备）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    api_url = f"https://api.figshare.com/v2/file/download/{file_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    last_error = ""
    for attempt in range(5):
        try:
            with requests.get(api_url, stream=True, timeout=(30, timeout), headers=headers,
                              allow_redirects=True) as resp:
                if resp.status_code == 202:
                    time.sleep(2)
                    continue
                resp.raise_for_status()
                total = 0
                with output_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                if total == 0:
                    raise RuntimeError("Downloaded 0 bytes")
            return True, ""
        except Exception as exc:
            last_error = str(exc)
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            if attempt < 4:
                time.sleep(2.0 * (attempt + 1))
                continue
    return False, last_error


def resolve_candidates(spec: Dict) -> List[str]:
    if spec.get("url"):
        return [str(spec["url"])]
    candidates: List[str] = []
    for hub in spec.get("hub_urls", []) or []:
        for filename in spec.get("file_candidates", []) or []:
            candidates.append(str(hub) + str(filename))
    return candidates


def _extract_sample_ids(matrix_path: Path, matrix_format: str) -> List[str]:
    if matrix_format not in {"csv", "tsv"}:
        return []
    sep = "," if matrix_format == "csv" else "\t"
    df = pd.read_csv(matrix_path, sep=sep, index_col=0, nrows=0)
    return [str(col) for col in df.columns]


def _write_sample_metadata(sample_ids: List[str], output_path: Path) -> Optional[Path]:
    if not sample_ids:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cell_line": ["" for _ in sample_ids],
            "condition": ["" for _ in sample_ids],
        }
    )
    df.to_csv(output_path, index=False)
    return output_path


def load_catalog(path: Optional[Path]) -> Dict:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_CATALOG


def save_catalog(path: Path, catalog: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=True, indent=2), encoding="utf-8")


def download_targets(target_specs: Iterable[Dict], out_dir: Path, dry_run: bool, timeout: int) -> Dict[str, Path]:
    target_paths: Dict[str, Path] = {}
    for spec in target_specs:
        name = str(spec.get("name", "target_map"))
        url = str(spec.get("url", ""))
        filename = str(spec.get("filename", ""))
        if not url or not filename:
            continue
        output_path = out_dir / "targets" / filename
        if dry_run:
            target_paths[name] = output_path
            continue
        ok, error = download_file(url, output_path, timeout=timeout)
        if ok:
            target_paths[name] = output_path
        else:
            print("Target map download failed:", name, error)
    return target_paths


def download_datasets(
    dataset_specs: Iterable[Dict],
    out_dir: Path,
    dry_run: bool,
    timeout: int,
    skip_existing: bool,
) -> Tuple[List[DownloadResult], Dict[str, Path]]:
    results: List[DownloadResult] = []
    dataset_paths: Dict[str, Path] = {}
    for spec in dataset_specs:
        name = str(spec.get("name", "dataset"))
        group = str(spec.get("group", "omics"))
        local_path = spec.get("local_path")
        if local_path:
            local = Path(local_path)
            if local.exists():
                dataset_paths[name] = local
                results.append(
                    DownloadResult(
                        name=name,
                        status="ok",
                        url=str(local),
                        path=str(local),
                        checksum_sha256=sha256_file(local),
                        retrieved_at_utc=_now_iso(),
                    )
                )
            else:
                results.append(
                    DownloadResult(
                        name=name,
                        status="failed",
                        url=str(local),
                        path=str(local),
                        error="local_path_not_found",
                        retrieved_at_utc=_now_iso(),
                    )
                )
            continue

        candidates = resolve_candidates(spec)
        output_dir = out_dir / group
        output_dir.mkdir(parents=True, exist_ok=True)
        status = "failed"
        error_text = ""
        used_url = ""
        # 从 URL 推导文件名，而非依赖 file_candidates
        first_url = candidates[0] if candidates else ""
        default_filename = Path(first_url).name if first_url else "dataset.tsv"
        output_path = output_dir / (spec.get("file_candidates", [default_filename])[0] if spec.get("file_candidates") else default_filename)

        if dry_run:
            used_url = candidates[0] if candidates else ""
            status = "skipped"
        else:
            # 检测 figshare URL 格式：使用 API 下载
            is_figshare = any("figshare.com/ndownloader/files/" in u for u in candidates)
            if is_figshare and candidates:
                figshare_url = candidates[0]
                file_id = figshare_url.rstrip("/").split("/")[-1]
                output_path = output_dir / file_id
                print(f"    [{name}] figshare file {file_id}...", flush=True)
                ok, err = download_figshare_file(file_id, output_path, timeout=timeout)
                if ok:
                    status = "ok"
                    dataset_paths[name] = output_path
                    print(f"    [{name}] OK (figshare)", flush=True)
                else:
                    error_text = err
                    status = "failed"
                    print(f"    [{name}] FAIL ({err[:80]})", flush=True)
            else:
                for idx, url in enumerate(candidates):
                    filename = Path(url).name
                    output_path = output_dir / filename
                    used_url = url
                    if skip_existing and output_path.exists():
                        status = "ok"
                        dataset_paths[name] = output_path
                        break
                    candidate_timeout = timeout if idx == len(candidates) - 1 else min(30, timeout)
                    print(f"    [{name}] trying {url[:100]}... (timeout={candidate_timeout}s)", flush=True)
                    ok, err = download_file(url, output_path, timeout=candidate_timeout)
                    if ok:
                        status = "ok"
                        dataset_paths[name] = output_path
                        print(f"    [{name}] OK <- {url[:100]}", flush=True)
                        break
                    error_text = err
                    print(f"    [{name}] FAIL ({err[:80]})", flush=True)
                    if idx == len(candidates) - 1:
                        status = "failed"

        checksum = sha256_file(output_path) if status == "ok" else ""
        results.append(
            DownloadResult(
                name=name,
                status=status,
                url=used_url,
                path=str(output_path),
                error=error_text,
                checksum_sha256=checksum,
                retrieved_at_utc=_now_iso(),
            )
        )

    return results, dataset_paths


def _read_text_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            yield line.rstrip("\n")


def extract_geo_series_matrix(raw_path: Path, output_path: Path) -> Optional[Path]:
    in_table = False
    rows: List[str] = []
    try:
        for line in _read_text_lines(raw_path):
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                rows.append(line)
    except Exception:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    if not rows:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output_path


def extract_cbioportal_tarball(raw_path: Path, output_dir: Path, filename: str) -> Optional[Path]:
    """从 cBioPortal tar.gz 归档中提取指定文件。"""
    import tarfile
    if not raw_path.exists():
        return None
    try:
        with tarfile.open(raw_path, "r:gz") as tar:
            for member in tar.getmembers():
                base = Path(member.name).name
                if base == filename:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / filename
                    with tar.extractfile(member) as src, output_path.open("wb") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                    return output_path
    except Exception:
        pass
    return None


def build_omics_manifest_entries(
    dataset_specs: Iterable[Dict],
    dataset_paths: Dict[str, Path],
    target_map_paths: Dict[str, Path],
    processed_dir: Path,
    write_sample_metadata: bool,
) -> List[Dict]:
    entries: List[Dict] = []
    for spec in dataset_specs:
        name = str(spec.get("name", "dataset"))
        matrix_path = dataset_paths.get(name)
        if not matrix_path:
            continue

        postprocess = str(spec.get("postprocess", "")).strip().lower()
        if postprocess == "geo_series_matrix":
            processed_name = str(spec.get("processed_filename") or f"{name}.tsv")
            processed_path = processed_dir / "geo" / processed_name
            if not processed_path.exists():
                extracted = extract_geo_series_matrix(matrix_path, processed_path)
                if extracted:
                    matrix_path = extracted
                else:
                    continue
            else:
                matrix_path = processed_path
        elif postprocess == "cbioportal_tarball":
            group_dir = processed_dir / str(spec.get("group", "tcga"))
            processed_name = str(spec.get("processed_filename") or "data.txt")
            processed_path = group_dir / processed_name
            if not processed_path.exists():
                extracted = extract_cbioportal_tarball(matrix_path, group_dir, processed_name)
                if extracted:
                    matrix_path = extracted
                else:
                    continue
            else:
                matrix_path = processed_path

        matrix_format = str(spec.get("matrix_format", "tsv"))
        sample_metadata_path = None
        if write_sample_metadata:
            try:
                sample_ids = _extract_sample_ids(matrix_path, matrix_format)
            except Exception:
                sample_ids = []
            if sample_ids:
                meta_path = processed_dir / "omics_sample_metadata" / f"{name}_samples.csv"
                sample_metadata_path = _write_sample_metadata(sample_ids, meta_path)

        target_map_name = spec.get("target_map_name")
        target_map_path = None
        if target_map_name:
            target_map_path = target_map_paths.get(str(target_map_name))

        entries.append(
            {
                "name": name,
                "modality": str(spec.get("modality", "expression")),
                "matrix_path": str(matrix_path),
                "matrix_format": matrix_format,
                "sample_metadata_path": str(sample_metadata_path) if sample_metadata_path else "",
                "sample_id_column": "sample_id",
                "cell_line_column": "cell_line",
                "condition_column": "condition",
                "min_value": float(spec.get("min_value", 0.0) or 0.0),
                "top_k_per_sample": int(spec.get("top_k_per_sample", 0) or 0),
                "source": str(spec.get("source", "unknown")),
                "version": str(spec.get("version", "unknown")),
                "confidence": float(spec.get("confidence", 0.7) or 0.7),
                "target_map_path": str(target_map_path) if target_map_path else "",
                "target_map_delimiter": str(spec.get("target_map_delimiter", "\t")),
            }
        )
    return entries


def prune_failed_catalog(catalog_path: Path, results: List[DownloadResult]) -> int:
    failed = {r.name for r in results if r.status == "failed"}
    if not failed:
        return 0
    catalog = load_catalog(catalog_path)
    if not isinstance(catalog, dict):
        return 0
    dataset_specs = [item for item in catalog.get("datasets", []) if item.get("name") not in failed]
    catalog["datasets"] = dataset_specs
    save_catalog(catalog_path, catalog)
    return len(failed)


def write_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download multi-omics lung cancer datasets")
    parser.add_argument("--catalog", default="data/raw/omics/omics_download_catalog.json", help="Catalog JSON path")
    parser.add_argument("--out-dir", default="data/raw/omics", help="Output directory for downloaded files")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/lung_cancer",
        help="Output directory for processed artifacts",
    )
    parser.add_argument(
        "--omics-manifest-out",
        default="data/processed/lung_cancer/omics_manifest.auto.json",
        help="Output omics manifest path",
    )
    parser.add_argument(
        "--download-manifest-out",
        default="data/raw/omics/omics_download_manifest.json",
        help="Output download manifest path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip actual downloads")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    parser.add_argument("--timeout", type=int, default=180, help="Download timeout in seconds")
    parser.add_argument("--no-sample-metadata", action="store_true", help="Do not generate sample metadata")
    parser.add_argument("--prune-failed", action="store_true", help="Remove failed datasets from catalog")
    parser.add_argument("--retry-after-prune", action="store_true", help="Re-run downloads after pruning failed datasets")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        save_catalog(catalog_path, DEFAULT_CATALOG)
        print("Catalog created at:", catalog_path)

    catalog = load_catalog(catalog_path)
    dataset_specs = catalog.get("datasets", []) if isinstance(catalog, dict) else []
    target_specs = catalog.get("target_maps", []) if isinstance(catalog, dict) else []

    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)

    target_paths = download_targets(target_specs, out_dir, args.dry_run, args.timeout)
    results, dataset_paths = download_datasets(
        dataset_specs,
        out_dir,
        args.dry_run,
        args.timeout,
        args.skip_existing,
    )

    if args.prune_failed:
        pruned = prune_failed_catalog(catalog_path, results)
        if pruned:
            print("Pruned failed datasets from catalog:", pruned)
            if args.retry_after_prune and not args.dry_run:
                catalog = load_catalog(catalog_path)
                dataset_specs = catalog.get("datasets", []) if isinstance(catalog, dict) else []
                results, dataset_paths = download_datasets(
                    dataset_specs,
                    out_dir,
                    args.dry_run,
                    args.timeout,
                    skip_existing=True,
                )

    omics_entries = build_omics_manifest_entries(
        dataset_specs,
        dataset_paths,
        target_paths,
        processed_dir,
        write_sample_metadata=not args.no_sample_metadata,
    )

    download_manifest = {
        "manifest_version": "1.0",
        "generated_at_utc": _now_iso(),
        "output_dir": str(out_dir),
        "datasets": [
            {
                **{
                    "name": r.name,
                    "status": r.status,
                    "url": r.url,
                    "path": r.path,
                    "error": r.error,
                    "checksum_sha256": r.checksum_sha256,
                    "retrieved_at_utc": r.retrieved_at_utc,
                },
                **{
                    "source": str(next((d.get("source") for d in dataset_specs if d.get("name") == r.name), "")),
                    "version": str(next((d.get("version") for d in dataset_specs if d.get("name") == r.name), "")),
                    "modality": str(next((d.get("modality") for d in dataset_specs if d.get("name") == r.name), "")),
                },
            }
            for r in results
        ],
    }

    write_manifest(Path(args.download_manifest_out), download_manifest)
    write_manifest(Path(args.omics_manifest_out), {"omics": omics_entries})

    ok_count = len([r for r in results if r.status == "ok"])
    print("Download completed: {}/{} datasets succeeded".format(ok_count, len(results)))
    print("Download manifest:", args.download_manifest_out)
    print("Omics manifest:", args.omics_manifest_out)


if __name__ == "__main__":
    main()

