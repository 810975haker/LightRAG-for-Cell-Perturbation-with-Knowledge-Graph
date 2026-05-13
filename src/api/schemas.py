from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentRequest(BaseModel):
    content: str
    metadata: Dict = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str


class MatrixIngestRequest(BaseModel):
    dataset_name: str
    matrix_path: str
    matrix_format: Literal["csv", "tsv", "mtx"]
    genes_path: Optional[str] = None
    cells_path: Optional[str] = None
    perturbation_path: Optional[str] = None
    delimiter: str = ","


class PredictRequest(BaseModel):
    cell_id: str
    perturbation: str
    question: str = ""
    target_genes: Optional[List[str]] = None
    top_k: int = 10
    top_n: Optional[int] = None
    max_hops: Optional[int] = None
    min_score: Optional[float] = None


class SMWContentRequest(BaseModel):
    page_title: str


class SMWExportRequest(BaseModel):
    wiki_page_title: str
    prediction: PredictRequest


class GraphFilterRequest(BaseModel):
    node_keyword: str = ""
    relation_keyword: str = ""
    node_query: str = ""
    relation_query: str = ""
    query_mode: Literal["and", "or"] = "or"
    match_mode: Literal["and", "or"] = "or"
    view_mode: Literal["stack", "replace"] = "stack"
    max_nodes: int = 120
    max_edges: int = 300


