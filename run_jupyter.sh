#!/bin/bash

# 设置代理环境变量
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

source /home/zql/code/lazy_slide/.venv/bin/activate
cd /home/zql/code/lazy_slide && uv run --with jupyter jupyter lab