from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    DocumentRequest,
    GraphFilterRequest,
    MatrixIngestRequest,
    PredictRequest,
    QueryRequest,
    SMWContentRequest,
)
from src.data.matrix_loader import CellPerturbDataset, MatrixLoader
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.rag.light_rag import LightRAG
from src.rag.vector_store import VectorStoreManager
from src.utils.helpers import process_smw_content
from src.utils.semantic_mediawiki import build_smw_template_payload
from config import KG_RANK_MAX_HOPS, KG_RANK_MIN_SCORE

router = APIRouter()

# 初始化组件
vector_store_manager = VectorStoreManager()
graph_builder = KnowledgeGraphBuilder()
graph_store = KnowledgeGraphStore()
light_rag = LightRAG(vector_store=vector_store_manager, graph_store=graph_store)
matrix_loader = MatrixLoader()
current_dataset: CellPerturbDataset | None = None
manifest_path = Path("data/raw/lung_cancer/download_manifest.json")


# 路由
@router.post("/documents/add")
async def add_document(request: DocumentRequest):
    """添加文档到向量存储"""
    try:
        document = vector_store_manager.create_document(
            request.content,
            request.metadata
        )
        light_rag.add_documents([document])
        return {"message": "文档添加成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_rag(request: QueryRequest):
    """使用Light-RAG查询"""
    try:
        result = light_rag.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/import_matrix")
async def import_matrix_dataset(request: MatrixIngestRequest):
    """从外部矩阵数据导入细胞扰动数据集。"""
    global current_dataset
    try:
        current_dataset = matrix_loader.load(
            matrix_path=request.matrix_path,
            matrix_format=request.matrix_format,
            dataset_name=request.dataset_name,
            genes_path=request.genes_path,
            cells_path=request.cells_path,
            perturbation_path=request.perturbation_path,
            delimiter=request.delimiter,
        )
        return {
            "message": "数据集导入成功",
            "dataset": current_dataset.dataset_name,
            "genes": int(current_dataset.expression.shape[0]),
            "cells": int(current_dataset.expression.shape[1]),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/knowledge-graph/build_from_dataset")
async def build_graph_from_dataset():
    """将已导入的表达矩阵构建成生物知识图谱。"""
    if current_dataset is None:
        raise HTTPException(status_code=400, detail="请先调用 /datasets/import_matrix 导入数据集")
    try:
        graph = graph_builder.build_from_dataset(current_dataset)
        graph_store.save_graph(graph)

        documents = []
        for record in current_dataset.perturbations:
            col = current_dataset.expression[record.cell_id]
            top_gene_names = col.sort_values(ascending=False).head(10).index.tolist()
            content = (
                f"dataset={current_dataset.dataset_name}; cell={record.cell_id}; "
                f"perturbation={record.perturbation}; top_genes={','.join(map(str, top_gene_names))}"
            )
            documents.append(
                vector_store_manager.create_document(
                    content,
                    {
                        "dataset": current_dataset.dataset_name,
                        "cell_id": record.cell_id,
                        "perturbation": record.perturbation,
                        "type": "cell_profile",
                    },
                )
            )
        light_rag.add_documents(documents)

        stats = graph_store.stats()
        return {
            "message": "知识图谱构建成功",
            "nodes": stats["entities"],
            "edges": stats["relations"],
            "documents_indexed": len(documents),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/build")
async def build_knowledge_graph(request: DocumentRequest):
    """从文本构建知识图谱"""
    try:
        graph = graph_builder.build_graph(request.content)
        graph_store.save_graph(graph)

        # 同时添加到向量存储
        document = vector_store_manager.create_document(
            request.content,
            {**request.metadata, "type": "knowledge_graph"}
        )
        light_rag.add_documents([document])

        return {
            "message": "知识图谱构建成功",
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smw/import")
async def import_from_smw(request: SMWContentRequest):
    """从Semantic MediaWiki导入内容"""
    try:
        content = process_smw_content(request.page_title)

        # 构建知识图谱
        graph = graph_builder.build_graph(content)
        graph_store.save_graph(graph)

        # 添加到向量存储
        document = vector_store_manager.create_document(
            content,
            {"source": "smw", "page_title": request.page_title}
        )
        light_rag.add_documents([document])

        return {
            "message": "SMW内容导入成功",
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(request: PredictRequest):
    """根据知识图谱+检索上下文进行细胞扰动预测。"""
    try:
        top_n = request.top_n if request.top_n is not None else request.top_k
        return light_rag.predict_perturbation(
            cell_id=request.cell_id,
            perturbation=request.perturbation,
            question=request.question,
            target_genes=request.target_genes,
            top_k=top_n,
            max_hops=request.max_hops if request.max_hops is not None else KG_RANK_MAX_HOPS,
            min_score=request.min_score if request.min_score is not None else KG_RANK_MIN_SCORE,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smw/prediction_payload")
async def prediction_payload_for_smw(request: PredictRequest):
    """返回可直接写入Semantic MediaWiki模板的文本。"""
    try:
        top_n = request.top_n if request.top_n is not None else request.top_k
        prediction = light_rag.predict_perturbation(
            cell_id=request.cell_id,
            perturbation=request.perturbation,
            question=request.question,
            target_genes=request.target_genes,
            top_k=top_n,
            max_hops=request.max_hops if request.max_hops is not None else KG_RANK_MAX_HOPS,
            min_score=request.min_score if request.min_score is not None else KG_RANK_MIN_SCORE,
        )
        return {
            "prediction": prediction,
            "smw_template": build_smw_template_payload(prediction),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/stats")
async def get_knowledge_graph_stats():
    """获取知识图谱统计信息"""
    try:
        return graph_store.stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/visualization")
async def get_knowledge_graph_visualization(max_nodes: int = 80, max_edges: int = 200):
    """返回用于前端图谱可视化的节点边数据。"""
    try:
        payload = graph_store.visualization_data(max_nodes=max_nodes, max_edges=max_edges)
        if not payload.get("nodes"):
            payload = graph_store.subgraph_data(
                node_keyword="",
                relation_keyword="",
                query_mode="or",
                view_mode="replace",
                max_nodes=max_nodes,
                max_edges=max_edges,
            )
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/filter-options")
async def get_knowledge_graph_filter_options(max_items: int = 200):
    """返回图谱查询下拉选项（节点与关系）。"""
    try:
        return graph_store.get_filter_options(max_items=max_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph/subgraph")
async def get_knowledge_graph_subgraph(request: GraphFilterRequest):
    """按节点/关系关键词筛选子图，供前端可视化检索。"""
    try:
        node_keyword = str(request.node_keyword or request.node_query or "")
        relation_keyword = str(request.relation_keyword or request.relation_query or "")
        query_mode = str(request.query_mode or request.match_mode or "or")
        return graph_store.subgraph_data(
            node_keyword=node_keyword,
            relation_keyword=relation_keyword,
            query_mode=query_mode,
            view_mode=request.view_mode,
            max_nodes=request.max_nodes,
            max_edges=request.max_edges,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph/neighbors")
async def get_knowledge_graph_neighbors(node_id: str, max_edges: int = 200):
    """返回指定节点1-hop邻居子图（用于前端点击展开）。"""
    try:
        if not str(node_id or "").strip():
            raise HTTPException(status_code=400, detail="node_id is required")
        return graph_store.one_hop_subgraph(node_id=node_id, max_edges=max_edges)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current_dataset")
@router.get("/knowledge-graph/datasets")
async def get_current_dataset():
    """返回当前图谱来源数据集概览（用于前端侧边栏）。"""
    try:
        stats = graph_store.stats()
    except Exception:
        stats = {"entities": 0, "relations": 0}

    payload = {
        "manifest_found": False,
        "manifest_version": None,
        "seed_genes": [],
        "active_dataset": current_dataset.dataset_name if current_dataset is not None else None,
        "datasets": [],
        "graph_stats": stats,
    }

    if not manifest_path.exists():
        return payload

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse manifest: {e}")

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in data.get("downloads", []):
        if item.get("status") != "ok":
            continue
        source = str(item.get("source") or "unknown")
        version = str(item.get("version") or "unknown")
        key = (source, version)
        if key not in grouped:
            grouped[key] = {
                "source": source,
                "version": version,
                "category_count": 0,
                "file_count": 0,
                "categories": set(),
            }
        grouped[key]["file_count"] = grouped[key]["file_count"] + 1
        categories = grouped[key]["categories"]
        category = str(item.get("category") or "unknown")
        if isinstance(categories, set):
            categories.add(category)

    datasets = []
    for value in grouped.values():
        categories = sorted(value["categories"]) if isinstance(value["categories"], set) else []
        datasets.append(
            {
                "source": value["source"],
                "version": value["version"],
                "file_count": value["file_count"],
                "category_count": len(categories),
                "categories": categories,
            }
        )
    datasets.sort(key=lambda it: (str(it["source"]).lower(), str(it["version"]).lower()))

    payload["manifest_found"] = True
    payload["manifest_version"] = data.get("manifest_version")
    payload["seed_genes"] = data.get("seed_genes", [])
    payload["datasets"] = datasets
    return payload


