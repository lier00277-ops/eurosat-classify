# 启动推理服务，浏览器打开 http://127.0.0.1:8001
# 8000 端口可能被其他程序占用，默认用 8001；也可以 .\scripts\run_serve.ps1 -Port 8002
param([int]$Port = 8001)

Set-Location (Join-Path $PSScriptRoot "..")
$env:EUROSAT_RESULTS = "D:\eurosat\results"
D:\eurosat\venv\Scripts\python.exe -m uvicorn src.serve:app --host 127.0.0.1 --port $Port
