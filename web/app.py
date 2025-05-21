#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import threading
import logging
import traceback
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from dotenv import load_dotenv
import io
from datetime import datetime
from contextlib import redirect_stdout

# 将项目根目录添加到模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 添加应用前缀环境变量支持
APPLICATION_ROOT = os.environ.get('APPLICATION_ROOT', '')

# 导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from newsmind_agent import NewsMindAgent
from log.logger import Logger

# 初始化Flask应用
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# 如果设置了应用前缀，则配置应用
if APPLICATION_ROOT:
    app.config['APPLICATION_ROOT'] = APPLICATION_ROOT
    # 增加以下配置用于处理静态文件的URL路径
    app.config['PREFERRED_URL_SCHEME'] = 'https'

# 配置日志
logger = logging.getLogger("NewsMindLogger")
logger.setLevel(logging.INFO)

# 添加 handler（避免重复添加）
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# 添加路径前缀处理器
@app.context_processor
def handle_url_prefix():
    def url_for_with_prefix(endpoint, **values):
        from flask import url_for
        if endpoint == 'static':
            # 如果是静态资源，且有应用前缀，则添加前缀路径
            url = url_for(endpoint, **values)
            if APPLICATION_ROOT and not url.startswith(APPLICATION_ROOT):
                url = f"{APPLICATION_ROOT}{url}"
            return url
        return url_for(endpoint, **values)
    return dict(url_for=url_for_with_prefix)

# 全局任务状态
current_task = None
task_output = []

# 自定义输出捕获类，用于存储输出而不是通过WebSocket发送
class OutputCapture(io.StringIO):
    def write(self, text):
        super().write(text)
        if text.strip():  # 只保存非空内容
            task_output.append(text)

# 路由：主页
@app.route('/')
def index():
    load_dotenv()  # 加载环境变量
    return render_template('index.html', application_root=APPLICATION_ROOT)

# 路由：开始任务
@app.route('/api/run', methods=['POST'])
def run_task():
    global current_task, task_output

    data = request.json
    emails = data.get('emails', [])
    user_input = data.get("user_query", "今天有哪些新闻？")  # 默认兜底
    
    if emails:
        os.environ['EMAIL_RECEIVER'] = ', '.join(emails)
    
    # 如果任务还在运行中，返回错误
    if current_task and current_task.is_alive():
        return jsonify({'success': False, 'message': '已有任务正在运行，请等待完成'})
    
    # 清空旧任务输出
    task_output.clear()

    # 启动任务线程
    def task_wrapper():
        global current_task
        try:
            run_news_aggregation_task(user_input)
        except Exception as e:
            print(f"[任务异常] {e}")
            traceback.print_exc()
        finally:
            print("[任务完成] 清理 current_task")
            current_task = None  # 线程结束，清理状态

    current_task = threading.Thread(target=task_wrapper)
    current_task.start()

    return jsonify({'success': True, 'message': '任务已启动'})



# 路由：获取任务状态
@app.route('/api/status', methods=['GET'])
def get_status():
    global current_task, task_output
    
    # 准备响应数据
    response = {
        'status': 'idle',
        'output': []
    }
    
    # 如果任务正在运行
    if current_task and current_task.is_alive():
        response['status'] = 'running'
    # 如果任务已完成
    elif current_task:
        response['status'] = 'completed'
    
    # 返回所有输出日志
    response['output'] = task_output
    
    return jsonify(response)


# 任务主函数（在单独的线程中运行）
def run_news_aggregation_task(user_input: str):
    """运行新闻聚合流程"""
    start_time = datetime.now()

    # 初始化 logger 并附加输出列表
    logger_instance = Logger()
    logger_instance.attach_task_output(task_output)

    logger_instance.info(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 开始新闻聚合流程")

    try:
        logger_instance.info(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 开始处理: 初始化智能体")

        agent = NewsMindAgent(
            use_collector=True,
            max_articles=5,
            clear_log=True
        )

        logger_instance.info(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 处理完成: 初始化智能体")
        
        agent.run(user_input)

        # 计算总耗时
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger_instance.info(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 新闻聚合流程完成 - 总耗时: {duration:.2f} 秒")
        
    except Exception as e:
        logger_instance.error(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 处理过程中出错: {str(e)}")
        logger.exception("任务执行出错")


# 启动应用
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 60005))
    app.run(host='0.0.0.0', port=port, debug=True) 