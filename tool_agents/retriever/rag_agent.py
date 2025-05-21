# rag_agent.py
import os
import json
import sys
import torch
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import (
    Settings,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from qdrant_client import QdrantClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tool_agents.base_agent import BaseAgent


# ===== Schema 定义 =====
@dataclass
class NewsItem:
    title: str
    link: str
    published: str
    summary: str
    domain: str
    search_keywords: List[str]
    language: str

    def to_metadata(self) -> Dict:
        return {
            "title": self.title,
            "link": self.link,
            "published": self.published,
            "summary": "",
            "domain": self.domain,
            "search_keywords": self.search_keywords,
            "language": self.language,
        }


# ===== RAG Agent 封装类 =====
class RAGAgent(BaseAgent):
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en",
        cache_folder: str = "models/hf_model_cache",
        chunk_size: int = 2048,
        top_k: int = 5,
    ):
        super().__init__()
        self.logger.info("[RAGAgent] 正在初始化...")
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.cache_folder = cache_folder
        self.index = None
        self.query_engine = None

        self._configure_embedding_model()
        self.logger.info("[RAGAgent] 初始化完成...")

    def _configure_embedding_model(self):
        embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model,
            cache_folder=self.cache_folder,
        )
        llm = HuggingFaceLLM(
            model_name="tiiuae/falcon-rw-1b",
            tokenizer_name="tiiuae/falcon-rw-1b",
            context_window=1024,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "do_sample": True},
            device_map="auto"
        )
        Settings.embed_model = embed_model
        self.logger.info(f"[RAGAgent] embedding_model初始化完成，embedding_model = {embed_model}")

        Settings.llm = llm
        self.logger.info(f"[RAGAgent] llm初始化完成，llm.model_name = {llm.model_name}")


    def load_news_items(self, raw: Dict[str, List[Dict]]) -> List[NewsItem]:
        """
        从按领域划分的字典解析文章，构造 NewsItem 实例列表。
        
        参数：
            raw: dict
                结构为 {domain: {[...],[...]}

        返回：
            List[NewsItem]
        """
        items = []
        for domain, articles in raw.items():
                for article in articles:
                    item = NewsItem(
                        title=article["title"],
                        link=article["link"],
                        published=article["published"],
                        summary=article["summary"],
                        domain=article["domain"],
                        search_keywords=article["search_keywords"],
                        language=article["language"],
                    )
                    items.append(item)
        return items

    def build_index(
        self,
        news_items: List[NewsItem],
        client_dir: str,
        vector_store_collection_name: str,
        save_dir: str,
        rebuild: bool = False  # 增加一个参数来控制是否重建
    ):
        docs = []
        for item in news_items:
            content = f"{item.domain} {' '.join(item.search_keywords)} {item.title}"
            metadata = item.to_metadata()
            doc = Document(text=content, metadata=metadata)
            docs.append(doc)

        splitter = SentenceSplitter(chunk_size=self.chunk_size)
        client = QdrantClient(path=client_dir)
        vector_store = QdrantVectorStore(client=client, collection_name=vector_store_collection_name)

        index = RAGAgent._create_index(docs, vector_store, splitter)
        index.storage_context.persist(persist_dir=save_dir)
        self.index = index
        self.query_engine = self._get_query_engine(index)

    @staticmethod
    def _create_index(
        docs: List[Document],
        vector_store: QdrantVectorStore,
        splitter: SentenceSplitter
    ):
        from llama_index.core import VectorStoreIndex


        return VectorStoreIndex.from_documents(
            docs,
            vector_store=vector_store,
            transformations=[splitter]
        )



    def load_index_from_local(self, storage_dir: str):
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        self.index = load_index_from_storage(storage_context)
        self.query_engine = self._get_query_engine(self.index)

    def _get_query_engine(self, index):
        return index.as_query_engine(similarity_top_k=self.top_k)

    def query(self, query_text: str) -> List[Dict]:
        if self.query_engine is None:
            raise ValueError("查询引擎未初始化，请先构建或加载索引。")
        response = self.query_engine.query(query_text)

        results = []
        for node in response.source_nodes:
            meta = node.node.metadata
            results.append({
                "title": meta.get("title"),
                "link": meta.get("link"),
                "published": meta.get("published"),
                "summary": meta.get("summary"),
                "domain": meta.get("domain"),
                "search_keywords": meta.get("search_keywords"),
                "language": meta.get("language")
            })
        return results

    def answer(self, docs: List[Dict], question: str) -> str:
        if not docs:
            self.logger.warning("未找到相关内容。")
            return "未找到相关内容。"

        context_text = "\n\n".join(
            f"标题: {doc['title']}" for doc in docs
        )

        prompt = (
            f"你是一名智能新闻助理。请根据以下内容回答用户的问题。\n\n"
            f"内容如下：\n{context_text}\n\n"
            f"用户问题：{question}\n\n"
            f"请用简洁、清晰的方式回答："
        )

        self.logger.info(f"生成的 Prompt:\n{prompt}")
        response = Settings.llm.complete(prompt)
        return str(response.text)


# ===== 示例调用 =====
if __name__ == "__main__":
    agent = RAGAgent()

    # 构建索引（仅首次）
    items = agent.load_news_items("data/json/news_collected.json")
    agent.build_index(
        news_items=items,
        client_dir="data/qdrant/client",
        vector_store_collection_name="news_collection",
        save_dir="data/qdrant/storage"
    )

    # 查询 & 回答
    agent.load_index_from_local("data/qdrant/storage")
    query_result = agent.query("AI in healthcare")

    agent.logger.info("=== 查询结果 ===")
    for r in query_result:  
        agent.logger.info(json.dumps(r, ensure_ascii=False, indent=2))

    answer = agent.answer(query_result, "AI in healthcare")
    agent.logger.info("=== 回答结果 ===")
    agent.logger.info(answer)
