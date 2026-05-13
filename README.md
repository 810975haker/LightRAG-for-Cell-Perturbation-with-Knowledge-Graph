# Light-RAG for Cell Perturbation with Knowledge Graph

This project implements a lightweight RAG pipeline for cell perturbation prediction:

- Import biological matrix datasets (`csv` / `tsv` / `mtx`)
- Build a perturbation-aware knowledge graph (cell-gene-perturbation)
- Build local retrieval index for supporting evidence
- Predict perturbation effects as gene-expression regression outputs (per-gene delta)
- Export Semantic MediaWiki-friendly prediction template payload

## KG Formalization

The virtual-cell KG is represented as `KG = (E, R, P)`:

- `E`: entities (`Cell`, `Gene`, `Protein`, `Pathway`, `Sample`, `Condition`, `miRNA`, etc.)
- `R`: directed relations between entities
- `P`: relation attributes, especially numeric strength signals

In implementation, each edge follows a weighted tuple style:

- `(e1, e2, T, p)` where `T` is relation type and `p` is weight-like property
- Persisted fields include `relation` (type), `weight`, `confidence`, `source`, `version`, `evidence`
- Direction-compatible fields `effect_sign` / `effect_label` are supported as edge properties (without changing relation type)
- Neo4j relationship type uses the normalized semantic relation (e.g., `ACTIVATES`, `EXPRESSES`); raw relation string is still stored in `relation`

Entity type normalization supports expanded types:

- `gene`
- `protein`
- `pathway`
- `Cell`
- `sample`
- `condition`
- `mirna`

Default deployment now uses Neo4j as graph backend (`KG_BACKEND=neo4j`), with local fallback if Neo4j is temporarily unavailable.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Flow (MVP)

1. Import matrix dataset
   - `POST /api/datasets/import_matrix`
2. Build KG + retrieval docs
   - `POST /api/knowledge-graph/build_from_dataset`
3. Predict perturbation
   - `POST /api/predict`
4. Get Semantic MediaWiki payload
   - `POST /api/smw/prediction_payload`

## Subgraph Query (KG Explore)

Endpoint: `POST /api/knowledge-graph/subgraph`

Request fields:

- `node_keyword` (recommended)
- `relation_keyword` (recommended)
- `query_mode`: `or | and`
- `view_mode`: `stack | replace` (used by frontend rendering strategy)
- `max_nodes`, `max_edges`

Compatibility aliases (still accepted):

- `node_query` -> `node_keyword`
- `relation_query` -> `relation_keyword`
- `match_mode` -> `query_mode`

`or` mode is conservative: only conditions that are actually filled by user participate in OR.

Response keeps stable `nodes` and `edges`, and adds optional `meta`:

```json
{
  "nodes": [],
  "edges": [],
  "meta": {
    "query": {
      "node_keyword": "EGFR",
      "relation_keyword": "inhibit",
      "query_mode": "or",
      "view_mode": "stack"
    },
    "match_stats": {
      "node_hit_edges": 18,
      "relation_hit_edges": 22,
      "both_hit_edges": 9,
      "returned_edges": 30
    }
  }
}
```

## Example Request

```powershell
curl -X POST http://127.0.0.1:8000/api/predict -H "Content-Type: application/json" -d '{"cell_id":"A549_cell_001","perturbation":"KO_EGFR","question":"预测EGFR扰动后的表达变化","target_genes":["EGFR","KRAS","TP53"],"top_n":6}'
```

## Lung Cancer KG Data Download

Download mixed biological sources (Reactome, STRING, Pathway Commons, NPInter, KEGG, WikiPathways, ENCODE eCLIP, BioGRID, NCBI Gene, Ensembl):

```powershell
python scripts/download_lung_cancer_kg_data.py
```

Custom output directory and seed genes:

```powershell
python scripts/download_lung_cancer_kg_data.py --out-dir data/raw/lung_cancer --genes EGFR KRAS TP53 ALK MET
```

`download_manifest.json` now records mixed-source provenance and parser metadata per dataset:

- `source`, `version`, `license`
- `parser`, `id_namespace`, `query_gene`
- `retrieved_at_utc`, `checksum_sha256`

Notes:

- Some upstream endpoints may intermittently fail or throttle; failed entries remain in manifest with `status=failed` for reproducibility.
- KEGG and WikiPathways are queried per seed gene (`--genes ...`) to keep download volume manageable.

## Multi-omics Data Download (Real Datasets)

Download public multi-omics matrices (TCGA LUAD/LUSC, CCLE, CPTAC when available) and build an omics manifest:

```powershell
python scripts/download_omics_data.py
```

The catalog is editable at `data/raw/omics/omics_download_catalog.json`. If any URLs fail, adjust `file_candidates` or `hub_urls` and re-run. The script writes:

- Download status manifest: `data/raw/omics/omics_download_manifest.json`
- Omics manifest for ingestion: `data/processed/lung_cancer/omics_manifest.auto.json`

If you want the catalog to remove failed sources and retry automatically:

```powershell
python scripts/download_omics_data.py --prune-failed --retry-after-prune --skip-existing
```

Use the generated omics manifest in KG build:

```powershell
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --omics-manifest data/processed/lung_cancer/omics_manifest.auto.json --replace
```

If your predictions show no reachable `protein::` nodes, download STRING bridge metadata (gene <-> protein mapping) and refresh manifest:

```powershell
python scripts/download_protein_bridge_data.py --raw-dir data/raw/lung_cancer --manifest data/raw/lung_cancer/download_manifest.json
```

By default, build also derives perturbation-oriented knowledge from seed genes:

- `pathway::perturbation::<METHOD>_<GENE>` nodes
- `pathway::perturb_method::<METHOD>` nodes
- edges for perturbation method, target gene, and affected neighbors

Default derived methods include `KO`, `KD`, `OE`, `CRISPRi`, `CRISPRa`, `RNAi`, `INHIBIT`, and support shared-pathway `2-hop` expansion.

You can tune or disable this step:

```powershell
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --max-derived-affected-per-seed 80
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --max-shared-pathway-2hop-per-seed 120
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --min-pathway-edge-strength 0.45 --min-shared-pathway-score 0.5
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --shared-pathway-keep-all
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --disable-perturbation-augmentation
```

For shared-pathway `2-hop`, edge `weight/confidence` is dynamically scaled by pathway evidence strength,
and weak candidates are denoised by `min_pathway_edge_strength` and `min_shared_pathway_score`.

KG loading policy:

- Neo4j edges are stored with typed relationships (e.g., `ACTIVATES`, `EXPRESSES`, `HAS_CNV`)
- Original interaction label is preserved in edge property `relation`
- Numeric edge strength is carried in `weight` and `confidence` (0..1)
- `evidence` stores both raw row text and structured fields (JSON string)
- Import is full-scale (no seed filtering); seed genes are marked as node property `is_seed=true`

Optional parser limits:

```powershell
python scripts/build_lung_cancer_kg.py --max-rows-per-source 50000 --min-string-score 750
```

Neo4j bulk import tuning (optional):

```dotenv
KG_NEO4J_BATCH_SIZE=3000
```

`save_graph` uses batched `UNWIND` writes for nodes and edges to speed up large imports.

Direction semantics policy (compatible mode):

- Keep semantic label in `relation` (even when relationship type is already typed)
- Add optional edge properties:
  - `effect_sign` (`-1` inhibit, `+1` activate, `0` unknown)
  - `effect_label` (human-readable tag)
- If `effect_sign` is missing, runtime falls back to relation-text heuristics

## Optional Protein Calibration (CCLE RPPA / CPTAC)

You can calibrate predicted protein abundance deltas using quantitative priors.

Build calibration table from a CCLE/CPTAC-like CSV/TSV:

```powershell
python scripts/build_protein_calibration.py --input data/raw/lung_cancer/ccle_rppa.csv --output data/processed/lung_cancer/protein_calibration.csv --source CCLE_RPPA
python scripts/build_protein_calibration.py --input data/raw/lung_cancer/cptac_proteomics.tsv --sep "`t" --output data/processed/lung_cancer/protein_calibration.csv --source CPTAC
```

Configure runtime:

```dotenv
PROTEIN_CALIBRATION_ENABLED=1
PROTEIN_CALIBRATION_PATH=./data/processed/lung_cancer/protein_calibration.csv
```

Template file is provided at `data/processed/lung_cancer/protein_calibration_template.csv`.

Validate loaded KG in Neo4j:

```powershell
python scripts/check_kg_in_neo4j.py --manifest data/raw/lung_cancer/download_manifest.json
```

If you only want warnings (no non-zero exit on failure):

```powershell
python scripts/check_kg_in_neo4j.py --manifest data/raw/lung_cancer/download_manifest.json --no-strict
```

## Semantic MediaWiki Front-end Integration

Built-in MediaWiki pages are now themed in A1 style via:

- `www/mediawiki/resources/custom/a1-home.css`
- `www/mediawiki/resources/custom/a1-home.js`

The script auto-renders three wiki pages by title:

- `Main_Page` (or `首页`) -> portal homepage with two entries
- `基因扰动预测` -> perturbation prediction page
- `知识图谱查询及可视化` -> KG query + visualization page with left dataset list

The KG page reads source/version list from:

- `GET /api/current_dataset` (recommended)
- `GET /api/knowledge-graph/datasets` (compatibility alias)

The KG query dropdown options are loaded from:

- `GET /api/knowledge-graph/filter-options`

Quick access URLs (local WAMP):

```powershell
http://localhost/mediawiki/index.php
http://localhost/mediawiki/index.php/基因扰动预测
http://localhost/mediawiki/index.php/图谱查询与可视化
http://localhost/mediawiki/index.php/知识图谱查询及可视化
```

You can add this to a MediaWiki page (`Common.js` or extension script) and call backend APIs:

```javascript
async function runPrediction() {
  const payload = { cell_id: "cellA", perturbation: "KO_GATA1", question: "预测影响" };
  const resp = await fetch("http://127.0.0.1:8000/api/smw/prediction_payload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  console.log(data.smw_template);
}
```

Or directly use the provided front-end sample: `frontend/smw_prediction_widget.js`.

## Local Demo Script

```powershell
python scripts/run_demo.py
```

## Tests

```powershell
python -m pytest -q
```

## Multi-omics KG Ingestion (Expression + Proteomics + CNV/Methylation + miRNA)

Use a JSON manifest to add omics matrices (expression, proteomics, CNV, methylation, miRNA). A small runnable sample is provided at `data/processed/lung_cancer/omics_manifest.sample.json`.

Example manifest schema (minimal):

```json
{
  "omics": [
    {
      "name": "tcga_luad_expression",
      "modality": "expression",
      "matrix_path": "data/raw/lung_cancer/omics/luad_expression.csv",
      "matrix_format": "csv",
      "sample_metadata_path": "data/raw/lung_cancer/omics/luad_samples.csv",
      "sample_id_column": "sample_id",
      "cell_line_column": "cell_line",
      "condition_column": "condition",
      "min_value": 0.2,
      "top_k_per_sample": 200,
      "source": "TCGA",
      "version": "2024Q4"
    }
  ]
}
```

Build KG with omics triples:

```powershell
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --omics-manifest data/processed/lung_cancer/omics_manifest.sample.json --replace
```

For large graphs or MemoryError, stream triples directly into Neo4j (skips NetworkX build):

```powershell
python scripts/build_lung_cancer_kg.py --manifest data/raw/lung_cancer/download_manifest.json --omics-manifest data/processed/lung_cancer/omics_manifest.auto.json --replace --streaming-neo4j
```

Quick demo (no Neo4j required):

```powershell
python scripts/run_omics_demo.py --manifest data/processed/lung_cancer/omics_manifest.sample.json
```

Supported omics relations (canonical):

- `expresses` (including miRNA expression)
- `has_protein_abundance`
- `has_cnv`
- `has_methylation`
- `belongs_to` (sample -> cell line)
- `has_condition`
- `regulates`
