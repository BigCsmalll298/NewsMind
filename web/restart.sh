#!/bin/bash

# 关闭60005端口的进程
sudo fuser -k 60005/tcp

# 设置应用前缀环境变量
export APPLICATION_ROOT=/autonews-agent

# 切换目录
cd "$(dirname "$0")/.."

echo "应用启动中，访问 https://tempshow.wenxuanhe.top/autonews-agent/"
echo "按Ctrl+C可以停止应用"

# 直接启动Flask应用
python web/app.py 