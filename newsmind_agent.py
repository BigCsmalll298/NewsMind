import os
import sys
import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from typing import Dict, List
from tool_agents.search import NewsSearchAgent
from tool_agents.integrator import IntegrationAgent
from tool_agents.mailer import EmailAgent
from tool_agents.retriever import RAGAgent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tool_agents.base_agent import BaseAgent

class NewsMindAgent(BaseAgent):
    """
    NewsMindAgent 作为系统的调度核心，负责协调新闻检索、内容整合、是否调用 RAG 与邮件发送。
    使用大模型 Function Call 决策是否使用 RAG 查询。
    """

    def __init__(self, use_collector: bool = False, max_articles: int = 20, clear_log: bool = False):
        super().__init__()
        self.logger.info(f"[NewsMindAgent] 正在初始化...")
        self.use_collector = use_collector
        self.max_articles = max_articles
        self.clear_log = clear_log

        # 初始化 Hugging Face 模型
        self.model_name = "deepseek-ai/deepseek-llm-7b-chat"
        self.logger.info(f"正在加载NewsMind的大脑llm：{self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        self.logger.info(f"NewsMind的大脑llm加载完成!")

        self.search_agent = NewsSearchAgent()
        self.rag_agent = RAGAgent()
        self.integrate_agent = IntegrationAgent()
        self.email_agent = EmailAgent()
        self.logger.info(f"[NewsMindAgent] 初始化完成")

        if self.clear_log:
            self._clear_log_file()

    def _clear_log_file(self):
        log_path = os.path.abspath(os.path.join("log", "logs", "NewsMind.log"))
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.truncate()
            self.logger.info(f"✅ 已清空日志文件: {log_path}")
        else:
            self.logger.info(f"❌ 日志文件不存在: {log_path}")

    def _llm_should_use_rag(self, user_query: str) -> bool:
        """
        用 Hugging Face 上的 DeepSeek 模型判断是否使用 RAG
        """
        prompt = f"""
        你是一个智能助手。请判断用户问题是否需要使用向量数据库（RAG）回答。
        只回答 True 或 False，且只能输出这两个单词之一，不要多余文字。

        规则：
        - 含“今天”、“最新”、“现在” -> False
        - 含“最近”、“自...以来” -> True

        示例：
        问：帮我查今天的新闻。 答：False
        问：帮我查最新的新闻。 答：False
        问：帮我查最近的科技新闻。 答：True
        问：帮我查自今年以来的财经新闻。 答：True

        用户的问题是：
        {user_query}

        回答：
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.8
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        self.logger.info(f"📨 DeepSeek 返回内容: {answer}")

        match = re.search(r'(\b\w+)\W*$', answer)
        if match:
            last_word = match.group(1)
        else:
            last_word = None

        self.logger.info(f"模型回答：{last_word}")
        if last_word == "True":
            return True
        elif last_word == "False":
            return False
        else:
            self.logger.warning(f"未知答案: {answer}, 默认False！！！")
            return False
        


    
    def group_by_domain(self, articles_list: List[Dict]) -> Dict[str, List[Dict]]:
        """
        将RAG的结果组织成字典以便后续整合
        """
        articles_by_domain = defaultdict(list)
        for article in articles_list:
            domain = article.get("domain", "unknown")
            articles_by_domain[domain].append(article)
        return dict(articles_by_domain)
    

    def count_total_dicts(self,data: Dict[str, List[dict]]) -> int:
        """
        统计文章数量
        """
        return sum(len(lst) for lst in data.values())

    def run(self, user_query: str):
        self.logger.info("🧠 [NewsMindAgent] 启动任务 ...")
        self.logger.info(f"📝 用户查询内容: {user_query}")

        articles = None #用来存储需要发送的文章

        # Step 1: 判断是否使用 RAG
        use_rag = self._llm_should_use_rag(user_query)
        if use_rag:
            self.logger.info("✅ 大模型判断结果：使用 RAG 查询")
            self.rag_agent.load_index_from_local("data/qdrant/storage")
            rag_answer = self.rag_agent.query(user_query)
            self.logger.info("🎯 RAG 查询结果：")
            for r in rag_answer:  
                self.logger.info(json.dumps(r, ensure_ascii=False, indent=2))
            articles = self.group_by_domain(rag_answer)
        else:
            self.logger.info("🚫 大模型判断结果：不使用 RAG")
            self.logger.info("🔍 开始新闻检索阶段 ...")
            unfiltered_articles, filtered_articles = self.search_agent.run(
                max_articles=self.max_articles,
                use_collector=self.use_collector
            )
            self.logger.info(f"🔎 检索到原始新闻数量: {self.count_total_dicts(unfiltered_articles)}，筛选后的新闻数量: {self.count_total_dicts(filtered_articles)}")
            articles = filtered_articles      

            #将最近搜索到的新闻向量追加存储到向量库
            self.logger.info("📦 开始构建向量索引 ...")
            items = self.rag_agent.load_news_items(unfiltered_articles)
            
            self.rag_agent.build_index(
                news_items=items,
                client_dir="data/qdrant/client",
                vector_store_collection_name="news_collection",
                save_dir="data/qdrant/storage"
            )
            self.logger.info("✅ 向量索引构建完成。")

        # Step 2: 整合内容
        self.logger.info("🧩 开始整合 HTML 邮件内容 ...")
        html_content = self.integrate_agent.integrate_content(
            filtered_articles=articles,
            max_articles=self.max_articles
        )
        self.logger.info("✅ 邮件内容整合完成。")

        # Step 3: 邮件发送
        self.logger.info("📨 开始发送邮件 ...")
        self.email_agent.send_email(html_content)
        self.logger.info("✅ 邮件发送完成，任务结束。")
