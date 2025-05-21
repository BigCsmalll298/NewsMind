import os
import json
import time
import random
import requests
import datetime
import feedparser
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tool_agents.base_agent import BaseAgent


class NewsCollector():
    def __init__(self, domainsConf_data = None, domains=None, news_api_key=None, seed=None, logger = None):
        """
        初始化搜索代理
        
        Args:
            news_api_key: 新闻API的密钥
            seed: 随机种子，用于复现结果
        """
        self.domainsConf_data = domainsConf_data
        self.domains = domains
        self.news_api_key = news_api_key 
        self.logger = logger
        self.news_api_key
               
        # 设置随机种子以便结果可复现
        if seed:
            random.seed(seed)

        self.logger.info("新闻收集器初始化完成！！！")

    def _collect_domain_news(self, domain: str, language: str, keyword_pairs: List[List[str]]) -> List[Dict]:
        """
        使用 Google News RSS 收集指定领域和语言的新闻

        Args:
            domain: 领域名称
            language: 'en' 或 'zh'
            keyword_pairs: 关键词对列表

        Returns:
            List[Dict]: 收集到的新闻文章
        """
        collected_articles = []

        # 语言区域参数设置
        language_map = {
            'en': ('en', 'US'),  # language_code, region_code
            'zh': ('zh', 'CN'),
        }
        language_code, region_code = language_map.get(language, ('en', 'US'))

        for keyword_pair in keyword_pairs:
            # 构造查询字符串，例如：AI AND Machine Learning
            query = " AND ".join(keyword_pair)
            query_encoded = query.replace(" ", "+")
            rss_url = (
                f"https://news.google.com/rss/search"
                f"?q={query_encoded}&hl={language_code}-{region_code}"
                f"&gl={region_code}&ceid={region_code}:{language_code}"
            )

            # 解析 RSS
            feed = feedparser.parse(rss_url)

            if not feed.entries:
                self.logger.warning(f"未找到关键词对 {keyword_pair} 的新闻")
                continue

            for entry in feed.entries[:5]:  # 最多取5篇文章
                article = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'domain': domain,
                    'search_keywords': keyword_pair,
                    'language': language
                }
                self.logger.info(f"  - {article['title']}")
                collected_articles.append(article)

            # 防止请求过快
            time.sleep(1)

        return collected_articles

    def collect_news(self, save_path: str = "data/json/news_collected.json"):
        """
        收集各个领域的新闻并保存为 JSON 文件
        
        Args:
            save_path: 输出结果的保存路径（默认为 data/json/news_collected.json）
        
        Returns:
            Dict: 按领域分类的新闻文章(未处理)
        """
        all_collected_articles = {}  # 存储所有收集到的文章（未处理）

        for domain in self.domains:
            self.logger.info(f"正在收集{domain}领域的新闻...")
            domain_conf = self.domainsConf_data[domain]
            
            # 收集英文新闻
            en_articles = self._collect_domain_news(domain, "en", domain_conf["keywords"]["en"])
            
            # 收集中文新闻
            zh_articles = self._collect_domain_news(domain, "zh", domain_conf["keywords"]["zh"])
            
            all_collected_articles[domain] = {
                "en": en_articles,
                "zh": zh_articles
            }

        self.logger.info("所有领域的新闻都已收集完成，准备进行后处理并保存为JSON文件。")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存为 JSON 文件，默认覆盖写入，每次只存储最近搜索到的新闻。便于查看数据格式。
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_collected_articles, f, ensure_ascii=False, indent=2)

        self.logger.info(f"新闻数据(未处理)已保存至：{save_path}")
        return all_collected_articles

            
if __name__ == "__main__":
    import argparse     # 解析命令行参数
    parser = argparse.ArgumentParser(description="搜索新闻")
    parser.add_argument("--clear-log", action="store_true", help="是否清空日志文件")
    args = parser.parse_args()
    if args.clear_log:
        log_path = os.path.abspath(os.path.join("log", "logs", "newsCollector.log"))
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.truncate()
            print(f"✅已清空日志文件: {log_path}")
        else:
            print(f"❌日志文件不存在: {log_path}")
    newsCollector = NewsCollector()
    newsCollector.collect_news()