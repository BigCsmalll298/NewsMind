import json
from typing import List, Dict, Set
import time
import dashscope
import os
from collections import Counter, defaultdict

class NewsProcessor:
    def __init__(self, domain_prompts: Dict, llm_api_key=None, model_name =None, logger = None):
        """
        初始化 NewsProcessor

        Args:
            domain_prompts: 每个领域的LLM提示词
            logger: 记录日志
        """
        self.domain_prompts = domain_prompts
        self.api_key = llm_api_key
        dashscope.api_key = self.api_key
        self.model_name = model_name
        self.logger = logger
        self.all_seen_urls: Set[str] = set()
        self.all_seen_titles: Set[str] = set()

        self.logger.info("新闻处理器配置完成！！！")
    
    def call_llm_api(self, prompt, temperature=0.7):
        """调用阿里云 DashScope 的大语言模型"""
        try:
            self.logger.debug(f"调用模型前 prompt: {prompt}")  
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                parameters={"temperature": temperature}
            )
            # self.logger.info("模型调用成功")
            # self.logger.info(f"模型输出: {response.output.text[:200]}...")  # 避免日志爆炸
            # self.logger.info("===============================================================")
            return response.output.text
        except Exception as e:
            self.logger.error(f"调用 LLM 模型失败: {e}")
            return ""
        
    def _remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """
        基于 URL或标题全局去重

        Args:
            articles: 原始文章列表

        Returns:
            List[Dict]: 去重后的文章
        """
        unique_articles = []
        for article in articles:
            url = article.get('link', '').strip()
            title = article.get('title', '').strip()
            if url and url not in self.all_seen_urls:
                self.all_seen_urls.add(url)
                unique_articles.append(article)
            elif not url and title and title not in self.all_seen_titles:
                self.all_seen_titles.add(title)
                unique_articles.append(article)
        return unique_articles
    
    def _format_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        格式化文章以便后续处理
        
        Args:
            articles: 文章列表
        
        Returns:
            List[Dict]: 格式化后的文章列表
        """
        formatted_articles = []
        for article in articles:
            source_info = article.get('source')
            source_name = (
                source_info.get('name', 'unknown') if isinstance(source_info, dict)
                else 'unknown'
            )

            formatted = {
                'title': article.get('title', ''),
                'summary': article.get('summary', ''),
                'content': article.get('summary', ''),
                'url': article.get('link', ''),
                'source': source_name,
                'publishedAt': article.get('published', ''),
                'domain': article.get('domain', ''),
                'language': '英文' if article.get('language') == 'en' else '中文',
                'keywords': article.get('search_keywords', [])
            }
            formatted_articles.append(formatted)
        return formatted_articles
    def _group_articles_by_domain(self, articles: List[Dict], domains: List[str]) -> Dict[str, List[Dict]]:
        """
        按领域组织文章，过滤只保留在给定领域中的文章。

        Args:
            articles: 所有文章列表
            domains: 有效的领域列表

        Returns:
            Dict[str, List[Dict]]: 按领域划分的文章
        """
        grouped_articles = defaultdict(list)
        for article in articles:
            domain = article.get('domain')
            if domain in domains:
                grouped_articles[domain].append(article)
        return grouped_articles


    def _filter_relevant_articles(self, articles: List[Dict], domains: List[str], domain_prompts:Dict, max_articles: int) -> List[Dict]:
        """
        使用LLM筛选最每个领域最相关的文章，准备经整合后发送邮件
        
        Args:
            articles: 候选文章列表
            domain: 领域名称列表
            max_articles: 返回的每个领域最大文章数量
        
        Returns:
            List[Dict]: 筛选后的文章列表
        """
        # each_domain_articles = Counter(article['domain'] for article in articles if 'domain' in article) 
        # print(each_domain_articles)

        #按领域组织文章
        unfiltered_domain_articles = self._group_articles_by_domain(articles, domains)


        # 按领域组织的摘要列表
        domain_summaries = defaultdict(list)  

        # 构建摘要：每个领域内编号从1开始
        for domain, domain_articles in unfiltered_domain_articles.items():
            for i, article in enumerate(domain_articles):
                title = article.get('title', 'No title')
                description = article.get('description', 'No description')
                summary = f"#{i+1}: {title}\n{description}\n"
                domain_summaries[domain].append(summary)

        filtered_domain_articles = defaultdict(list)

        for domain in domains:
            if domain not in domain_summaries:
                continue  # 跳过无摘要的领域

            # 构造提示语
            prompt = f"""
            {domain_prompts[domain]}
            
            以下是{domain}领域的新闻文章列表:
            
            {"".join(domain_summaries[domain])}
            
            请从上述列表中选择{max_articles}篇最重要和最相关的文章。
            选择时请考虑文章的多样性，覆盖不同的子领域和内容，避免选择过于相似的内容。
            请按重要性排序，仅返回文章编号，格式如下:
            [1, 5, 8, ...]
            """

            # 调用LLM API筛选文章
            response = self.call_llm_api(prompt, temperature=0.3)
            response_text = response.strip()

            # 提取JSON数组
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            json_str = response_text[start_idx:end_idx]
            selected_indices = json.loads(json_str)

            # 验证并调整索引（基于当前domain的文章列表）
            domain_articles = unfiltered_domain_articles[domain]
            valid_indices = []
            for idx in selected_indices:
                adjusted_idx = idx - 1  # 1-based 转换为 0-based
                if 0 <= adjusted_idx < len(domain_articles):
                    valid_indices.append(adjusted_idx)

            # 选择文章并加入结果
            for i in valid_indices[:max_articles]:
                filtered_domain_articles[domain].append(domain_articles[i])

        # 返回结构化结果
        return filtered_domain_articles
    
    def process_articles(self, 
                     articles: List[Dict], 
                     domains: List[str], 
                     domain_prompts: Dict, 
                     max_articles: int = 5) -> Dict[str, List[Dict]]:
        """
        执行文章的完整处理流程：去重、格式化、领域筛选

        Args:
            articles: 所有待处理的原始文章
            domains: 需要筛选的领域列表
            domain_prompts: 每个领域的提示词
            max_articles: 每个领域保留的最大文章数

        Returns:
            Dict[str, List[Dict]]: 每个领域筛选出的文章
        """
        self.logger.info("开始处理文章...")
        
        removed_articles = self._remove_duplicates(articles)
        self.logger.info(f"全局去重后剩余文章数: {len(removed_articles)}")

        formatted_articles = self._format_articles(removed_articles)
        self.logger.info(f"文章格式化完成，共{len(formatted_articles)}篇")

        self.logger.info("开始调用大模型进行文章筛选...")
        filtered_articles = self._filter_relevant_articles(
            articles=formatted_articles,
            domains=domains,
            domain_prompts=domain_prompts,
            max_articles=max_articles
        )
        
        # 日志打印每个领域筛选结果
        for domain, domain_articles in filtered_articles.items():
            self.logger.info(f"【{domain}】领域筛选出{len(domain_articles)}篇文章：")
            # for i, article in enumerate(domain_articles):
            #     title = article.get('title', 'No title')
            #     desc = article.get('description', '')[:80].replace('\n', ' ')
            #     # self.logger.info(f"  - 第{i+1}篇: {title} | 摘要前80字: {desc}...")

        self.logger.info("文章处理流程完成。")
        return filtered_articles







