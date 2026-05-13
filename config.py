import os
from dotenv import load_dotenv

load_dotenv()

# 基础配置
PROJECT_NAME = "Light-RAG Knowledge Graph System"
VERSION = "1.0.0"
DEBUG = True

# 科大讯飞配置（可选）
IFLYTEK_API_KEY = os.getenv("IFLYTEK_API_KEY", "")
IFLYTEK_LLM_MODEL = os.getenv("IFLYTEK_LLM_MODEL", "x1")
IFLYTEK_EMBEDDING_MODEL = os.getenv("IFLYTEK_EMBEDDING_MODEL", "text-embedding")

# 向量存储配置
VECTOR_STORE_PATH = "./vector_store"
EMBEDDING_DIMENSION = 1536

# 知识图谱配置
KG_BACKEND = os.getenv("KG_BACKEND", "neo4j")  # neo4j | networkx
KG_HOST = os.getenv("KG_HOST", "localhost")
KG_PORT = int(os.getenv("KG_PORT", "7474"))
KG_USERNAME = os.getenv("KG_USERNAME", "neo4j")
KG_PASSWORD = os.getenv("KG_PASSWORD", "password")
KG_NEO4J_BATCH_SIZE = int(os.getenv("KG_NEO4J_BATCH_SIZE", "2000"))

# 图谱排序与去噪配置
KG_RANK_MAX_HOPS = int(os.getenv("KG_RANK_MAX_HOPS", "3"))
KG_RANK_MIN_SCORE = float(os.getenv("KG_RANK_MIN_SCORE", "0.01"))
KG_RANK_DISTANCE_DECAY = float(os.getenv("KG_RANK_DISTANCE_DECAY", "0.78"))
KG_RANK_PATHWAY_BOOST = float(os.getenv("KG_RANK_PATHWAY_BOOST", "1.12"))

# 蛋白丰度校准（可选）：用于接入CCLE RPPA/CPTAC等定量蛋白数据
PROTEIN_CALIBRATION_PATH = os.getenv("PROTEIN_CALIBRATION_PATH", "./data/processed/lung_cancer/protein_calibration.csv")
PROTEIN_CALIBRATION_ENABLED = os.getenv("PROTEIN_CALIBRATION_ENABLED", "1") == "1"

# 生物数据导入配置
DEFAULT_MATRIX_DELIMITER = os.getenv("DEFAULT_MATRIX_DELIMITER", ",")
KG_EXPR_THRESHOLD = float(os.getenv("KG_EXPR_THRESHOLD", "0.5"))
KG_TOP_GENES_PER_CELL = int(os.getenv("KG_TOP_GENES_PER_CELL", "20"))

# Semantic MediaWiki 配置
SMW_URL = os.getenv("SMW_URL", "http://localhost/mediawiki/api.php")
SMW_USERNAME = os.getenv("SMW_USERNAME", "admin")
SMW_PASSWORD = os.getenv("SMW_PASSWORD", "password")


def iflytek_enabled() -> bool:
	return ":" in IFLYTEK_API_KEY
