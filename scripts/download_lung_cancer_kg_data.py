from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus

import pandas as pd
import requests


DEFAULT_GENES = [
    "EGFR",
    "KRAS",
    "ALK",
    "MET",
    "TP53",
    "PIK3CA",
    "BRAF",
    "ERBB2",
    "ROS1",
    "RET",
]


def resolve_kegg_gene_ids(symbol: str, timeout: int = 30) -> List[str]:
    gene_symbol = str(symbol or "").strip().upper()
    if not gene_symbol:
        return []
    try:
        url = "https://rest.kegg.jp/find/genes/{}".format(quote_plus(gene_symbol))
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        ids: List[str] = []
        for line in resp.text.splitlines():
            if "\t" not in line:
                continue
            left, right = line.split("\t", 1)
            kegg_id = left.strip()
            if not kegg_id.startswith("hsa:"):
                continue
            desc = right.upper()
            # Prefer exact symbol mention to reduce false matches.
            if re.search(r"(^|[\s,;]){}([\s,;]|$)".format(re.escape(gene_symbol)), desc):
                ids.append(kegg_id)
        if ids:
            return list(dict.fromkeys(ids))
    except Exception:
        pass
    return []


def download_file(url: str, output_path: Path, timeout: int = 120) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"url": url, "path": str(output_path), "status": "ok", "error": ""}
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with output_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    return result


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sources(seed_genes: List[str]) -> List[Dict]:
    # These are public download endpoints; availability may vary by mirror/time.
    sources: List[Dict] = [
        {
            "name": "reactome_pathway_relations",
            "category": "pathway",
            "source": "Reactome",
            "version": "current",
            "license": "Reactome license",
            "url": "https://reactome.org/download/current/ReactomePathwaysRelation.txt",
            "filename": "ReactomePathwaysRelation.txt",
            "parser": "reactome_pathway_relations",
            "id_namespace": "reactome",
        },
        {
            "name": "reactome_pathways",
            "category": "pathway",
            "source": "Reactome",
            "version": "current",
            "license": "Reactome license",
            "url": "https://reactome.org/download/current/ReactomePathways.txt",
            "filename": "ReactomePathways.txt",
            "parser": "reactome_pathways",
            "id_namespace": "reactome",
        },
        {
            "name": "reactome_uniprot_pathway",
            "category": "pathway",
            "source": "Reactome",
            "version": "current",
            "license": "Reactome license",
            "url": "https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt",
            "filename": "UniProt2Reactome_All_Levels.txt",
            "parser": "reactome_uniprot_pathway",
            "id_namespace": "uniprot_reactome",
        },
        {
            "name": "string_human_ppi",
            "category": "protein_protein_interaction",
            "source": "STRING",
            "version": "v12.0",
            "license": "CC BY 4.0",
            "url": "https://stringdb-static.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
            "filename": "9606.protein.links.v12.0.txt.gz",
            "parser": "string_ppi",
            "id_namespace": "string",
        },
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
        {
            "name": "pathway_commons_hgnc_sif",
            "category": "gene_gene_interaction",
            "source": "Pathway Commons",
            "version": "v14",
            "license": "Source database licenses apply",
            "url": "https://download.baderlab.org/PathwayCommons/PC2/v14/PathwayCommons14.All.hgnc.sif.gz",
            "filename": "PathwayCommons14.All.hgnc.sif.gz",
            "parser": "pathway_commons_hgnc_sif",
            "id_namespace": "hgnc_symbol",
        },
        {
            "name": "npinter_rna_interaction",
            "category": "rna_protein_regulation",
            "source": "NPInter",
            "version": "v4",
            "license": "NPInter terms",
            "url": "http://bigdata.ibp.ac.cn/npinter4/download/file/interaction_NPInterv4.txt.gz",
            "filename": "interaction_NPInterv4.txt.gz",
            "parser": "npinter",
            "id_namespace": "mixed",
        },
        {
            "name": "biogrid_human_interactions",
            "category": "gene_gene_interaction",
            "source": "BioGRID",
            "version": "4.4.231",
            "license": "BioGRID terms",
            "url": "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.231/BIOGRID-ORGANISM-4.4.231.tab3.zip",
            "filename": "BIOGRID-ORGANISM-4.4.231.tab3.zip",
            "parser": "biogrid_tab3",
            "id_namespace": "gene_symbol",
        },
        {
            "name": "ncbi_human_gene_info",
            "category": "gene_annotation",
            "source": "NCBI Gene",
            "version": "current",
            "license": "NCBI disclaimer",
            "url": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
            "filename": "Homo_sapiens.gene_info.gz",
            "parser": "ncbi_gene_info",
            "id_namespace": "ncbi_geneid",
        },
        {
            "name": "ensembl_human_gtf",
            "category": "gene_annotation",
            "source": "Ensembl",
            "version": "release-114",
            "license": "Ensembl data reuse policy",
            "url": "https://ftp.ensembl.org/pub/release-114/gtf/homo_sapiens/Homo_sapiens.GRCh38.114.gtf.gz",
            "filename": "Homo_sapiens.GRCh38.114.gtf.gz",
            "parser": "ensembl_gtf",
            "id_namespace": "ensembl_gene",
        },
        {
            "name": "encode_eclip_report",
            "category": "rna_protein_regulation",
            "source": "ENCODE",
            "version": "current",
            "license": "ENCODE terms",
            "url": "https://www.encodeproject.org/report.tsv?type=Experiment&assay_title=eCLIP&status=released&limit=all",
            "filename": "encode_eclip_experiment_report.tsv",
            "parser": "encode_eclip_report",
            "id_namespace": "gene_symbol",
        },
    ]

    for gene in seed_genes:
        gene_key = gene.strip().upper()
        if not gene_key:
            continue
        kegg_ids = resolve_kegg_gene_ids(gene_key)
        if not kegg_ids:
            # Backward-compatible fallback; may return empty for symbol-only queries.
            kegg_ids = ["hsa:{}".format(gene_key)]
        for kegg_id in kegg_ids:
            kegg_token = kegg_id.replace(":", "_")
            sources.append(
                {
                    "name": "kegg_gene_pathway_{}_{}".format(gene_key, kegg_token),
                    "category": "pathway",
                    "source": "KEGG",
                    "version": "current",
                    "license": "KEGG terms",
                    "url": "https://rest.kegg.jp/link/pathway/{}".format(quote_plus(kegg_id)),
                    "filename": "kegg_pathway_{}_{}.txt".format(gene_key, kegg_token),
                    "parser": "kegg_gene_pathway",
                    "id_namespace": "kegg_gene_pathway",
                    "query_gene": gene_key,
                }
            )
        sources.append(
            {
                "name": "wikipathways_gene_query_{}".format(gene_key),
                "category": "pathway",
                "source": "WikiPathways",
                "version": "current",
                "license": "CC0",
                "url": "https://webservice.wikipathways.org/findPathwaysByText?query={}&species=Homo+sapiens&format=json".format(
                    quote_plus(gene_key)
                ),
                "filename": "wikipathways_{}.json".format(gene_key),
                "parser": "wikipathways_query_json",
                "id_namespace": "wikipathways",
                "query_gene": gene_key,
            }
        )

    return sources


def save_seed_gene_list(genes: List[str], base_dir: Path) -> Path:
    gene_file = base_dir / "lung_cancer_seed_genes.csv"
    pd.DataFrame({"gene": genes}).to_csv(gene_file, index=False)
    return gene_file


def main():
    parser = argparse.ArgumentParser(description="Download lung-cancer-related KG raw data sources")
    parser.add_argument("--out-dir", default="data/raw/lung_cancer", help="Output directory")
    parser.add_argument("--genes", nargs="*", default=DEFAULT_GENES, help="Seed genes for downstream filtering")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_version": "1.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "strategy": "mixed_sources",
        "seed_genes": args.genes,
        "downloads": [],
    }

    for source in build_sources(args.genes):
        category_dir = out_dir / source["category"]
        output_path = category_dir / source["filename"]
        info = download_file(source["url"], output_path)
        if info["status"] == "ok":
            info["checksum_sha256"] = sha256_file(output_path)
        else:
            info["checksum_sha256"] = ""
        info.update(
            {
                "name": source["name"],
                "category": source["category"],
                "source": source["source"],
                "version": source["version"],
                "license": source.get("license", ""),
                "parser": source.get("parser", ""),
                "id_namespace": source.get("id_namespace", ""),
                "query_gene": source.get("query_gene", ""),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        manifest["downloads"].append(info)

    seed_file = save_seed_gene_list(args.genes, out_dir)
    manifest["seed_gene_file"] = str(seed_file)

    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    ok_count = len([d for d in manifest["downloads"] if d["status"] == "ok"])
    print("Download completed: {}/{} sources succeeded".format(ok_count, len(manifest["downloads"])))
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()

