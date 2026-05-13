from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_path: Path, timeout: int = 180) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = {"url": url, "path": str(output_path), "status": "ok", "error": ""}
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with output_path.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
        info["checksum_sha256"] = sha256_file(output_path)
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = str(exc)
        info["checksum_sha256"] = ""
    return info


def sources() -> List[Dict]:
    return [
        {
            "name": "signor_human_signed_relations",
            "category": "gene_gene_interaction",
            "source": "SIGNOR",
            "version": "current",
            "license": "SIGNOR terms",
            "url": "https://signor.uniroma2.it/getData.php?organism=9606&format=csv",
            "filename": "SIGNOR_human_signed.csv",
            "parser": "signor_signed_relations",
            "id_namespace": "gene_protein_mixed",
        },
        {
            "name": "omnipath_signed_interactions",
            "category": "gene_gene_interaction",
            "source": "OmniPath",
            "version": "current",
            "license": "OmniPath terms",
            "url": "https://omnipathdb.org/interactions?format=tsv",
            "filename": "omnipath_interactions.tsv",
            "parser": "omnipath_signed_relations",
            "id_namespace": "gene_symbol",
        },
    ]


def upsert_download(manifest: Dict, item: Dict) -> None:
    downloads = manifest.setdefault("downloads", [])
    for idx, old in enumerate(downloads):
        if old.get("name") == item.get("name"):
            downloads[idx] = item
            return
    downloads.append(item)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download signed regulatory data and prune failed entries")
    parser.add_argument("--raw-dir", default="data/raw/lung_cancer")
    parser.add_argument("--manifest", default="data/raw/lung_cancer/download_manifest.json")
    parser.add_argument("--keep-failed", action="store_true", help="Keep failed entries in manifest")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "manifest_version": "1.3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(raw_dir),
            "strategy": "mixed_sources",
            "downloads": [],
        }

    for src in sources():
        out_path = raw_dir / src["category"] / src["filename"]
        info = download_file(src["url"], out_path)
        info.update(
            {
                "name": src["name"],
                "category": src["category"],
                "source": src["source"],
                "version": src["version"],
                "license": src.get("license", ""),
                "parser": src.get("parser", ""),
                "id_namespace": src.get("id_namespace", ""),
                "query_gene": "",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        upsert_download(manifest, info)

    if not args.keep_failed:
        manifest["downloads"] = [item for item in manifest.get("downloads", []) if item.get("status") == "ok"]

    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Manifest updated:", manifest_path)
    print("Downloads kept:", len(manifest.get("downloads", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

