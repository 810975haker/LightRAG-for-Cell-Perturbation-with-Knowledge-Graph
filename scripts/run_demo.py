import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.matrix_loader import CellPerturbDataset, CellPerturbRecord
from src.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from src.knowledge_graph.graph_store import KnowledgeGraphStore
from src.rag.light_rag import LightRAG
from src.rag.vector_store import VectorStoreManager



def main():
    expression = pd.DataFrame(
        {
            "cellA": [1.2, 0.2, 2.4],
            "cellB": [0.1, 1.8, 0.3],
        },
        index=["GATA1", "TP53", "MYC"],
    )
    dataset = CellPerturbDataset(
        dataset_name="demo",
        expression=expression,
        perturbations=[
            CellPerturbRecord(cell_id="cellA", perturbation="KO_GATA1"),
            CellPerturbRecord(cell_id="cellB", perturbation="OE_TP53"),
        ],
    )

    graph_builder = KnowledgeGraphBuilder()
    graph_store = KnowledgeGraphStore()
    vector_store = VectorStoreManager()
    rag = LightRAG(vector_store=vector_store, graph_store=graph_store)

    graph = graph_builder.build_from_dataset(dataset)
    graph_store.save_graph(graph)

    docs = []
    for record in dataset.perturbations:
        docs.append(
            vector_store.create_document(
                content="cell={} perturbation={} top_genes={}".format(
                    record.cell_id,
                    record.perturbation,
                    ",".join(dataset.expression[record.cell_id].sort_values(ascending=False).head(3).index),
                ),
                metadata={"cell_id": record.cell_id, "perturbation": record.perturbation},
            )
        )
    rag.add_documents(docs)

    prediction = rag.predict_perturbation("cellA", "KO_GATA1", "预测主要受影响通路")
    print("Demo prediction:")
    print(prediction)


if __name__ == "__main__":
    main()

