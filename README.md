# AI相关
### 本地python环境
- `/opt/third/etsai_python`

### 主程序目录
- `/opt/third/etsai`
### 程序启动
 - AI主逻辑        `/bin/etsai`
 - 对外接口服务     `/bin/etsai_api`
 - web页面         `/bin/etsai_web`

### 数据存储路径
- sqlite `/opt/third/.local/etsai`
- 图片存储路径 `/opt/third/.local/etsai/images`


```shell
├── README.md
├── __pycache__
│   ├── get_ip.cpython-310.pyc
│   ├── get_ip.cpython-38.pyc
│   └── utils.cpython-38.pyc
├── api_flask.py                  对外提供接口服务
├── app.py                        web页面
├── bin                           启动
│   ├── etsai
│   ├── etsai_api
│   └── etsai_web
├── flow.py                       ai主逻辑基于jina-flow
├── flow_cred.py
├── init.d                        开机启动程序
│   ├── S99ai
│   ├── S99ai_api
│   └── S99ai_web
├── input_face.py                 tmp
├── install_dep.sh                安装依赖
├── query_face.py                 tmp
├── requirements.txt
├── tmp                           临时图片文件夹
│   ├── search_tmp1.jpg
│   ├── yt-003.jpg
│   ├── yt-006.jpg
│   └── yt_right.jpg
├── torch                         模型
│   └── checkpoints
│       └── 20180402-114759-vggface2.pt
└── utils.py                      工具类
```
