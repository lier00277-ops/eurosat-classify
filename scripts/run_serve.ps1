# 启动推理服务，浏览器打开 http://127.0.0.1:8000
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $PSScriptRoot "..")
$env:EUROSAT_RESULTS = "D:\eurosat\results"
D:\eurosat\venv\Scripts\python.exe -m uvicorn src.serve:app --host 127.0.0.1 --port 8000
