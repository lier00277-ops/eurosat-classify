# 环境变量：缓存、数据、虚拟环境都放在 D 盘。
# 使用：. .\scripts\env.ps1

$env:UV_INDEX_URL   = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:UV_CACHE_DIR   = "D:\eurosat\cache\uv"
$env:PIP_CACHE_DIR  = "D:\eurosat\cache\pip"
$env:HF_HOME        = "D:\eurosat\cache\hf"
$env:HF_DATASETS_CACHE = "D:\eurosat\cache\hf\datasets"
$env:EUROSAT_DATA   = "D:\eurosat\data"

# 激活 D 盘虚拟环境
$env:VIRTUAL_ENV = "D:\eurosat\venv"
$env:PATH = "D:\eurosat\venv\Scripts;" + $env:PATH
