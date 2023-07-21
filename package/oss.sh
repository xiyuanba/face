#!/bin/bash

# 定义帮助信息函数
usage() {
  echo "Usage: $0 [option]"
  echo "Options:"
  echo "  upload    Upload the file to OSS"
  echo "  download  Download the file from OSS"
}

# 检查命令行参数
if [ $# -ne 1 ]; then
  usage
  exit 1
fi

# 根据命令行参数执行相应的操作
if [ "$1" = "download" ]; then
  ossutil -c /root/.ossutilconfig cp oss://etsme-config/config/cookie/releases/packages.json ./packages.json
  echo "Download completed!"
elif [ "$1" = "upload" ]; then
  ossutil -c /root/.ossutilconfig cp ./packages.json oss://etsme-config/config/cookie/releases/packages.json
  echo "Upload completed!"
else
  usage
  exit 1
fi
