# 把本项目相关的“安装物/缓存/数据”全部指向 D 盘，避免占用 C 盘。
# 用法：在 PowerShell 中执行  . .\scripts\env.ps1   （注意前面有个点，表示“点源”到当前会话）

$env:UV_INDEX_URL   = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:UV_CACHE_DIR   = "D:\eurosat\cache\uv"
$env:PIP_CACHE_DIR  = "D:\eurosat\cache\pip"
$env:HF_HOME        = "D:\eurosat\cache\hf"
$env:HF_DATASETS_CACHE = "D:\eurosat\cache\hf\datasets"
$env:EUROSAT_DATA   = "D:\eurosat\data"

# 激活 D 盘虚拟环境
$env:VIRTUAL_ENV = "D:\eurosat\venv"
$env:PATH = "D:\eurosat\venv\Scripts;" + $env:PATH
