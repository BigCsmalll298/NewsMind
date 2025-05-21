import os
import sys
import json
from typing import List, Dict, Set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tool_agents.base_agent import BaseAgent  
from tool_agents.search.news_collector import NewsCollector
from tool_agents.search.news_processor import NewsProcessor
from tool_agents.search.domainsConf_generator import DomainsConfGenerator



class NewsSearchAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.collector = None
        self.processor = None
        self.domain_keywords = {}
        self.domain_prompts = {}

        self.configurator = DomainsConfGenerator(
            model_name=self.search_agent_llm_model_name,
            api_key=self.search_agent_llm_api_key,
            logger=self.logger
        )
        self.logger.info("[NewsSearchAgent]领域配置器初始化完成！！！")

        self._load_domain_config()

    def _load_domain_config(self):
        conf_path = os.path.join("data", "json", "domainsConf.json")
        if not os.path.exists(conf_path):
            self.logger.warning("未找到 domainsConf.json，尝试自动生成...")
            self.configurator.generate_domainsConf_json(output_path=conf_path)
            self.logger.info("领域配置生成完成！！！")

        with open(conf_path, 'r', encoding='utf-8') as f:
            self.domainsConf_data = json.load(f)

        if not self.domainsConf_data:
            raise ValueError("domainsConf.json 中没有任何领域定义")

        self.domains = []
        self.domain_keywords = {}
        self.domain_prompts = {}

        for domain, info in self.domainsConf_data.items():
            if "keywords" not in info or "prompt" not in info:
                raise ValueError(f"领域 {domain} 缺少 'keywords' 或 'prompt'")
            self.domains.append(domain)
            self.domain_keywords[domain] = info["keywords"]
            self.domain_prompts[domain] = info["prompt"]
        self.logger.info(f"[NewsSearchAgent] 成功加载领域配置：{self.domains}")

    def init_collector(self):
        self.collector = NewsCollector(
            domainsConf_data=self.domainsConf_data,
            domains=self.domains,
            news_api_key=self.news_api_key,
            seed=None,
            logger=self.logger
        )

    def init_processor(self):
        self.processor = NewsProcessor(
            domain_prompts=self.domain_prompts,
            llm_api_key=self.search_agent_llm_api_key,
            model_name=self.search_agent_llm_model_name,
            logger=self.logger
        )

    def load_collected_articles(self) -> List[Dict]:
        """
        从 JSON 文件加载所有领域、所有语言的文章，并扁平化为一个列表

        Returns:
            List[Dict]: 所有文章组成的统一列表
        """
        file_path = os.path.join("data", "json", "news_collected.json")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        all_articles = []
        for domain_articles in raw_data.values():  # 遍历每个领域
            for lang_articles in domain_articles.values():  # 遍历每个语言
                all_articles.extend(lang_articles)  # 合并文章列表

        return all_articles
    def run(self, max_articles: int, use_collector: bool = False):
        """
        主运行流程，包括可选采集、文章加载、按领域组织、过滤等操作。
        """
        self.logger.info("[NewsSearchAgent] 正在启动运行流程...")

        if use_collector:
            self.logger.info("使用新闻采集器进行新闻采集...")
            self.init_collector()
            self.collector.collect_news()

        self.logger.info("加载已采集的新闻文章...")
        articles = self.load_collected_articles()

        self.logger.info("初始化新闻处理器...")
        self.init_processor()

        self.logger.info("按领域组织未筛选文章，用于向量化或RAG...")
        unfiltered_domain_articles = self.processor._group_articles_by_domain(
            articles=articles,
            domains=self.domains
        )

        self.logger.info("进行文章筛选处理，准备传送给聚合代理...")
        filtered_domain_articles = self.processor.process_articles(
            articles=articles,
            domains=self.domains,
            domain_prompts=self.domain_prompts,
            max_articles=max_articles
        )
        self.logger.info("✅ [NewsSearchAgent] 运行完成！")
        return unfiltered_domain_articles, filtered_domain_articles



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="新闻搜索代理")
    parser.add_argument("--clear-log", action="store_true", help="是否清空日志文件")
    parser.add_argument("--use-collector", action="store_true", help="是否启用新闻采集器进行采集")
    args = parser.parse_args()

    if args.clear_log:
        log_path = os.path.abspath(os.path.join("log", "logs", "NewsSearch_agent.log"))
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.truncate()
            print(f"✅已清空日志文件: {log_path}")
        else:
            print(f"❌日志文件不存在: {log_path}")

    agent = NewsSearchAgent()
    unfiltered, filtered = agent.run(max_articles=5, use_collector=args.use_collector)
    # 打印 unfiltered 的第一个领域的前三篇文章
    print("\n🔍 [Unfiltered] 第一个领域的前三篇文章内容：")
    first_domain = next(iter(unfiltered))  # 获取第一个领域名称
    for i, article in enumerate(unfiltered[first_domain][:3]):
        print(f"\n--- 第 {i+1} 篇文章 ---")
        for key, value in article.items():
            print(f"{key}: {value}")

    # 打印 filtered 的第一个领域的前三篇文章
    print("\n✅ [Filtered] 第一个领域的前三篇文章内容：")
    first_domain_filtered = next(iter(filtered))
    for i, article in enumerate(filtered[first_domain_filtered][:3]):
        print(f"\n--- 第 {i+1} 篇文章 ---")
        for key, value in article.items():
            print(f"{key}: {value}")










