import os
import sys
import json
import random
import dashscope
from typing import List, Dict, Set
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from tool_agents.base_agent import BaseAgent
from jinja2 import Environment, FileSystemLoader 

class IntegrationAgent(BaseAgent):
    """内容整合智能体"""
    
    def __init__(self):
        """初始化内容整合智能体"""
        super().__init__()
        dashscope.api_key = self.search_agent_llm_api_key
        self.model_name = self.search_agent_llm_model_name
        self.inspirational_quotes = [
            "创新是区分领导者和跟随者的特质。 - 史蒂夫·乔布斯",
            "知识就是力量。 - 弗朗西斯·培根",
            "学习是一种永无止境的旅程。 - 不详",
            "成功不是最终的，失败也不是致命的，重要的是继续前进的勇气。 - 温斯顿·丘吉尔",
            "每一个不曾起舞的日子，都是对生命的辜负。 - 尼采",
            "世上最遥远的距离不是生与死，而是我站在你面前，你却不知道我爱你。 - 泰戈尔",
            "生活中最重要的事情是要有一个远大的目标，并有决心去实现它。 - 约翰·洛克菲勒",
            "科技的进步在于取代那些能够被理性描述的工作。 - 埃隆·马斯克",
            "发现创新不等于创新，创新不等于创业。 - 吴军",
            "每一个伟大的事业都始于一个不合理的假设。 - 彼得·泰尔"
        ]
        self.logger.info("[IntegrationAgent]初始化完成！")
        

    def call_llm_api(self, title_prompt, temperature=0.7):
        """调用阿里云 DashScope 的大语言模型"""
        self.logger.info("[IntegrationAgent]正在调用大模型生成邮件标题...")
        try:
            self.logger.debug(f"调用模型前 prompt: {title_prompt}")  
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=title_prompt,
                parameters={"temperature": temperature}
            )
            self.logger.info("模型调用成功")
            self.logger.info(f"模型输出: {response.output.text[:200]}...")  # 避免日志爆炸
            self.logger.info("===============================================================")
            return response.output.text
        except Exception as e:
            self.logger.error(f"调用 LLM 模型失败: {e}")
            return ""
    
    def gene_email_title(self, filtered_articles: Dict[str, List[Dict]]) -> str:
        """
        调用大模型生成邮件标题
        """
        title_prompt = f"""你是一个专业的新闻编辑，请为今天的新闻摘要生成一个简洁有力的标题。
            新闻内容如下:{filtered_articles},你需要根据里面的内容提炼标题。
            标题应该能吸引读者注意力，并且反映当天的主要新闻内容。
            请不要超过30个字，只需要返回标题本身，不要加任何其他内容。"""
        email_title = self.call_llm_api(title_prompt=title_prompt, temperature=0.7)
        return email_title

    def integrate_content(self, filtered_articles: Dict[str, List[Dict]], max_articles: int) -> str:
        """
        按领域整合新闻内容并生成邮件内容

        Args:
            filtered_articles: 经过llm筛选后的按领域组织的文章字典
            max_articles: 最大文章数

        Returns:
            str: 整合后的邮件内容
        """
        self.logger.info("[IntegrationAgent]正在加载邮件模板...")
        env = Environment(loader=FileSystemLoader("tool_agents/integrator"))
        template = env.get_template("email_template.html")
        self.logger.info("[IntegrationAgent]邮件模板加载完成!")
        
        self.logger.info("[IntegrationAgent]正在渲染邮件内容...")
        random_quote = random.choice(self.inspirational_quotes)
        today = datetime.now().strftime("%Y年%m月%d日")
        email_title = self.gene_email_title(filtered_articles)
        email_html = template.render(
            title=email_title,
            filtered_articles=filtered_articles, 
            max_articles=max_articles,
            random_quote=random_quote,
            date = today
        )
        self.logger.info("[IntegrationAgent]新闻内容聚合完成，准备发送邮件!")
        return email_html
                


        
