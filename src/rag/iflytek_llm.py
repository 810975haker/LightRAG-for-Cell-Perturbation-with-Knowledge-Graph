import base64
import hashlib
import time
from typing import List, Optional

import requests
from langchain.embeddings.base import Embeddings
from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field

from config import IFLYTEK_API_KEY


class IFlytekEmbeddings(Embeddings):
    """科大讯飞文本嵌入模型"""

    def __init__(self, api_key: str = IFLYTEK_API_KEY):
        self.api_key = api_key
        self.app_id, self.api_secret = self._parse_api_key(api_key)
        self.embeddings_url = "https://maas-api.cn-huabei-1.xf-yun.com/v2/embeddings"

    def _parse_api_key(self, api_key: str) -> tuple:
        """解析API密钥，格式为app_id:api_secret"""
        parts = api_key.split(":")
        if len(parts) != 2:
            raise ValueError("API密钥格式错误，应为app_id:api_secret")
        return parts[0], parts[1]

    def _get_auth_headers(self, uri: str) -> dict:
        """生成认证头信息"""
        # 获取当前时间戳
        cur_time = str(int(time.time()))

        # 生成签名
        sha = hashlib.sha256()
        sha.update((uri + cur_time + self.api_secret).encode('utf-8'))
        checksum = sha.digest()
        checksum_base64 = base64.b64encode(checksum).decode('utf-8')

        return {
            "X-Appid": self.app_id,
            "X-CurTime": cur_time,
            "X-CheckSum": checksum_base64,
            "Content-Type": "application/json"
        }

    def embed_query(self, text: str) -> List[float]:
        """获取单个文本的嵌入向量"""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """获取多个文本的嵌入向量"""
        # 构造请求数据
        data = {
            "input": [{"text": text} for text in texts]
        }

        # 获取认证头
        headers = self._get_auth_headers("/v2/embeddings")

        # 发送请求
        response = requests.post(self.embeddings_url, headers=headers, json=data)
        result = response.json()

        # 处理响应
        if result.get("code") != 0:
            raise Exception(f"嵌入失败: {result.get('message', '未知错误')}")

        # 提取嵌入向量
        embeddings = []
        for item in result.get("data", []):
            embedding = item.get("embedding", [])
            embeddings.append(embedding)

        return embeddings


class IFlytekLLM(LLM):
    """科大讯飞大语言模型"""

    # 使用Pydantic Field定义类属性
    api_key: str = Field(default=IFLYTEK_API_KEY)
    model_name: str = Field(default="chatglm3")

    @property
    def _llm_type(self) -> str:
        """返回LLM类型"""
        return "iflytek"

    def __init__(self, **kwargs):
        # 调用父类的__init__方法
        super().__init__(**kwargs)
        # 解析API密钥
        self.app_id, self.api_secret = self._parse_api_key(self.api_key)
        self.llm_url = "https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions"

    def _parse_api_key(self, api_key: str) -> tuple:
        """解析API密钥，格式为app_id:api_secret"""
        parts = api_key.split(":")
        if len(parts) != 2:
            raise ValueError("API密钥格式错误，应为app_id:api_secret")
        return parts[0], parts[1]

    def _get_auth_headers(self, uri: str) -> dict:
        """生成认证头信息"""
        # 获取当前时间戳
        cur_time = str(int(time.time()))

        # 生成签名
        sha = hashlib.sha256()
        sha.update((uri + cur_time + self.api_secret).encode('utf-8'))
        checksum = sha.digest()
        checksum_base64 = base64.b64encode(checksum).decode('utf-8')

        return {
            "X-Appid": self.app_id,
            "X-CurTime": cur_time,
            "X-CheckSum": checksum_base64,
            "Content-Type": "application/json"
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None) -> str:
        """生成文本（使用LangChain兼容的_call方法）"""
        # 构造请求数据
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # 获取认证头
        headers = self._get_auth_headers("/v2/chat/completions")

        # 发送请求
        response = requests.post(self.llm_url, headers=headers, json=data)
        result = response.json()

        # 处理响应
        if result.get("code") != 0:
            raise Exception(f"生成失败: {result.get('message', '未知错误')}")

        # 提取生成的文本
        message = result.get("choices", [])[0].get("message", {})
        content = message.get("content", "")

        return content

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager=None) -> LLMResult:
        """生成文本（符合LangChain 1.x版本要求）"""
        generations = []

        for prompt in prompts:
            content = self._call(prompt, stop, run_manager)
            generation = Generation(text=content)
            generations.append([generation])

        return LLMResult(generations=generations)
