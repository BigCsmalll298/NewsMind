import os
import sys
from dotenv import load_dotenv
import requests
from abc import ABC, abstractmethod
import re
import dashscope
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from log.logger import Logger

# 指定 .env 文件路径并加载环境变量
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", ".env"))
load_dotenv(dotenv_path=env_path)

class BaseAgent(ABC):
    """基础智能体类"""
    """
    统一获取.env中的各种配置，传递给不同子类。
    """
    def __init__(self):
        """初始化基础智能体"""
        self.search_agent_llm_api_key = os.getenv("DASHSCOPE_API_KEY") #大模型的api_key
        self.search_agent_llm_model_name = os.getenv("MODEL_NAME")  # 如 qwen-turbo, qwen-plus 等
        self.openai_api_key=os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.smtp_server = os.getenv("SMTP_SERVER")
        self.smtp_port = int(os.getenv("SMTP_PORT", "465"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.email_receiver = os.getenv("EMAIL_RECEIVER", "")
        self.logger = Logger(log_filename="NewsMind.log")
    
    def is_traditional_chinese(self, text):
        """检测文本是否包含繁体中文
        
        Args:
            text (str): 需要检测的文本
            
        Returns:
            bool: 是否包含繁体中文
        """
        # 常见的简体-繁体对应字符
        trad_chars = {
            '髮': '发', '壹': '一', '貳': '二', '參': '三', '肆': '四',
            '為': '为', '這': '这', '說': '说', '對': '对', '時': '时',
            '從': '从', '會': '会', '來': '来', '學': '学', '國': '国',
            '與': '与', '產': '产', '內': '内', '係': '系', '點': '点',
            '實': '实', '發': '发', '經': '经', '關': '关', '樣': '样',
            '單': '单', '歲': '岁', '們': '们', '區': '区', '衝': '冲',
            '東': '东', '車': '车', '話': '话', '過': '过', '億': '亿',
            '預': '预', '當': '当', '體': '体', '麼': '么', '電': '电',
            '務': '务', '開': '开', '買': '买', '總': '总', '問': '问',
            '門': '门', '見': '见', '認': '认', '隻': '只', '飛': '飞',
            '處': '处', '專': '专', '將': '将', '書': '书', '號': '号',
            '長': '长', '應': '应', '變': '变', '節': '节', '義': '义',
            '連': '连', '錢': '钱', '場': '场', '馬': '马', '顯': '显',
            '親': '亲', '顧': '顾', '語': '语', '頭': '头', '條': '条',
            '鐘': '钟', '鳥': '鸟', '龍': '龙', '齊': '齐'
        }
        
        # 检查文本中是否包含繁体字符
        for char in text:
            if char in trad_chars:
                return True
        
        # 使用正则表达式匹配一些繁体中文特有的Unicode范围
        # 繁体中文常见于Unicode中的一些特定范围
        pattern = r'[\u4E00-\u9FFF]'  # 基本汉字范围
        matches = re.findall(pattern, text)
        
        # 如果匹配到的汉字超过一定数量，并且包含一些典型的繁体字符组合
        if len(matches) > 5:
            trad_patterns = ['這個', '時間', '國家', '經濟', '發展', '關於', '實現', '東西', '學習', '電話']
            for pattern in trad_patterns:
                if pattern in text:
                    return True
        
        return False