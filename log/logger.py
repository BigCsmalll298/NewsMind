import logging
import os

class Logger:
    def __init__(self, log_dir="log/logs", log_filename=None):
        # 获取项目根路径（logger.py 的上上级目录）
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 设置日志文件夹完整路径
        self.log_dir = os.path.join(root_dir, log_dir)
        self.log_filename = log_filename or "NewsMind.log"
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志文件的完整路径
        log_filepath = os.path.join(self.log_dir, self.log_filename)

        # ✅ 使用固定 logger 名称，确保不会重复创建不同的 logger 实例
        self.logger = logging.getLogger("NewsMindLogger")
        self.logger.setLevel(logging.DEBUG)

        # ✅ 关键：防止重复添加 handler
        if not self.logger.handlers:
            # 文件 handler
            file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)

            # 控制台 handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加 handler
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(f"🔍 {message}")

    def info(self, message):
        self.logger.info(f"✅ {message}")

    def warning(self, message):
        self.logger.warning(f"⚠️ {message}")

    def error(self, message):
        self.logger.error(f"❌ {message}")

    def critical(self, message):
        self.logger.critical(f"🚨 {message}")

    def attach_task_output(self, output_list):
        """附加日志输出到 task_output 列表"""
        handler = TaskOutputLogHandler(output_list)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)



class TaskOutputLogHandler(logging.Handler):
    def __init__(self, output_list):
        super().__init__()
        self.output_list = output_list

    def emit(self, record):
        log_entry = self.format(record)
        if log_entry.strip():
            self.output_list.append(log_entry)
