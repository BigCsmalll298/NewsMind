import os
import json
import argparse
import dashscope   
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from dotenv import load_dotenv
# from log.logger import Logger

# 指定 .env 文件路径并加载环境变量
# env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", ".env"))
# load_dotenv(dotenv_path=env_path)

class DomainsConfGenerator:
    """
    生成新闻搜索的领域词、领域对应的关键词对以及筛选提示词
    """
    def __init__(self, model_name=None, api_key = None, logger = None):
        self.logger = logger        # 初始化 Logger
        self.model_name = model_name         # 读取模型配置与 API key
        self.api_key = api_key
        dashscope.api_key = self.api_key

        self.logger.info(f"[NewsSearchAgent]大模型名称设置为: {self.model_name}")
        self.logger.info(f"[NewsSearchAgent]大模型_API_KEY 是否加载: {'是' if self.api_key else '否'}")

    def call_llm_api(self, prompt, temperature=0.7):
        """调用阿里云 DashScope 的大语言模型"""
        try:
            self.logger.debug(f"调用模型前 prompt: {prompt}")  
            response = dashscope.Generation.call(
                model=self.model_name,
                prompt=prompt,
                parameters={"temperature": temperature}
            )
            self.logger.info("模型调用成功")
            self.logger.debug(f"模型输出: {response.output.text[:200]}...")  # 避免日志爆炸
            return response.output.text
        except Exception as e:
            self.logger.error(f"调用 LLM 模型失败: {e}")
            return ""

    def get_domains(self):
        prompt = """
            请参考新浪新闻、腾讯新闻、网易新闻等门户网站，列出常见的新闻领域（英文标识符 + 中文名称）。用如下JSON格式输出：
            {
            "technology": "科技",
            "economy": "经济",
            "sports": "体育",
            "entertainment": "娱乐",
            "..."
            }
            只输出标准 JSON，不要添加任何说明或注释。
        """.strip()
        resp = self.call_llm_api(prompt)
        try:
            domains = json.loads(resp)
            self.logger.info(f"获取到的领域信息: {domains}")
            return domains
        except Exception as e:
            self.logger.error(f"解析新闻领域失败: {e}")
            return {}

    def get_domain_data(self, domain, zh_name):
        prompt = f"""
            你是一个新闻关键词生成助手。请根据以下领域，生成对应关键词和文章筛选提示。

            - 领域英文标识符: {domain}
            - 领域中文名: {zh_name}

            请返回以下结构的JSON, 不要添加其他说明：

            {{
            "keywords": {{
                "en": [["英文关键词1", "英文关键词2"], ["英文关键词3", "英文关键词4"], ["英文关键词5", "英文关键词6"], ["英文关键词7", "英文关键词8"]],
                "zh": [["中文关键词1", "中文关键词2"], ["中文关键词3", "中文关键词4"], ["中文关键词5", "中文关键词6"], ["中文关键词7", "中文关键词8"]]
            }},
            "prompt": "你是一位资深的{zh_name}新闻筛选专家。请选择最重要、最具影响力的{zh_name}新闻文章。优先考虑与{zh_name}领域最新动态、趋势走向、行业影响力事件相关的内容。请确保所选文章具有代表性、深度分析价值，并能体现{zh_name}发展的重大方向与潜在影响。"
            }}
        """.strip()
        resp = self.call_llm_api(prompt)
        try:
            data = json.loads(resp)
            self.logger.info(f"解析成功：{domain} ({zh_name})")
            return data
        except Exception as e:
            self.logger.error(f"解析 {domain} 的关键词结构失败: {e}")
            return {}

    def generate_domainsConf_json(self, output_path="data/json/domainsConf.json"):
        all_data = {}
        domains = self.get_domains()
        for domain, zh_name in domains.items():
            self.logger.info(f"生成领域中: {domain} ({zh_name})")
            result = self.get_domain_data(domain, zh_name)
            if result:
                all_data[domain] = result
            else:
                self.logger.warning(f"跳过领域: {domain}, 原因：无生成结果")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"关键词生成完毕，已写入文件: {output_path}")
        except Exception as e:
            self.logger.error(f"写入关键词 JSON 文件失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="领域配置生成器")
    parser.add_argument("--clear-log", action="store_true", help="是否清空日志文件")
    args = parser.parse_args()

    if args.clear_log:
        log_path = os.path.abspath(os.path.join("log", "logs", "domainsConf.log"))
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.truncate()
            print(f"✅已清空日志文件: {log_path}")
        else:
            print(f"❌日志文件不存在: {log_path}")

    generator = DomainsConfGenerator()
    generator.generate_domainsConf_json()

