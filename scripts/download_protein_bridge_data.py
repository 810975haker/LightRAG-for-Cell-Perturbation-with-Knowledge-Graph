from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

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
            "name": "string_human_protein_info",
            "category": "protein_protein_interaction",
            "source": "STRING",
            "version": "v12.0",
            "license": "CC BY 4.0",
            "url": "https://stringdb-static.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
            "filename": "9606.protein.info.v12.0.txt.gz",
            "parser": "string_protein_info",
            "id_namespace": "string",
        },
        {
            "name": "string_human_protein_aliases",
            "category": "protein_protein_interaction",
            "source": "STRING",
            "version": "v12.0",
            "license": "CC BY 4.0",
            "url": "https://stringdb-static.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
            "filename": "9606.protein.aliases.v12.0.txt.gz",
            "parser": "string_protein_aliases",
            "id_namespace": "string",
        },
    ]


def upsert_download(manifest: Dict, item: Dict) -> None:
    downloads = manifest.setdefault("downloads", [])
    for idx, old in enumerate(downloads):
        if old.get("name") == item.get("name"):
            downloads[idx] = item
            return
    downloads.append(item)


def main():
    parser = argparse.ArgumentParser(description="Download STRING bridge data for gene->protein connectivity")
    parser.add_argument("--raw-dir", default="data/raw/lung_cancer")
    parser.add_argument("--manifest", default="data/raw/lung_cancer/download_manifest.json")
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

    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    ok_count = len([d for d in manifest.get("downloads", []) if d.get("status") == "ok"])
    print("Bridge download done. successful entries in manifest:", ok_count)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

