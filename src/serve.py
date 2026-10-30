"""FastAPI 推理服务。启动：uvicorn src.serve:app --port 8001"""
from __future__ import annotations

import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from src.infer import load_model, predict_pil

app = FastAPI(title="EuroSAT 地物分类服务", version="1.0")

# 启动时加载模型
MODEL, CLASSES, MEAN, STD, DEVICE = load_model()
print(f"[serve] model loaded: {len(CLASSES)} classes on {DEVICE}")


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EuroSAT 地物分类</title>
<style>
  body { font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif; max-width: 760px;
         margin: 40px auto; padding: 0 16px; color: #222; background: #fbfdff; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: #778; font-size: 13px; margin-bottom: 22px; }
  .drop { border: 2px dashed #7aa; border-radius: 12px; padding: 30px; text-align: center;
          color: #667; cursor: pointer; transition: .2s; display: block; background: #fff; }
  .drop:hover, .drop.over { border-color: #39a; background: #f2fbff; }
  #preview { max-width: 240px; border-radius: 10px; margin-top: 16px; display: none; }
  #result { margin-top: 18px; }
  .bar { height: 10px; background: #e6eef2; border-radius: 6px; overflow: hidden; margin: 4px 0 10px; }
  .bar > i { display: block; height: 100%; background: linear-gradient(90deg, #39a, #5cc); }
  .badge { display: inline-block; background: #39a; color: #fff; border-radius: 999px;
           padding: 3px 12px; font-weight: 600; }
  .small { color: #778; font-size: 13px; }
  code { background: #eef4f7; padding: 1px 6px; border-radius: 5px; }
</style>
</head>
<body>
  <h1>EuroSAT 遥感地物识别</h1>
  <div class="sub">上传一张卫星 / 航拍图像，返回 10 类地物中最可能的类别。模型：手写 ResNet18 · 测试集准确率 97.61%</div>
  <label class="drop" id="drop">
    点击选择图片，或把图片拖到这里
    <input type="file" id="file" accept="image/*" hidden>
  </label>
  <img id="preview" alt="">
  <div id="result"></div>
<script>
  const drop = document.getElementById('drop');
  const file = document.getElementById('file');
  const preview = document.getElementById('preview');
  const result = document.getElementById('result');

  function render(html) { result.innerHTML = html; }

  async function show() {
    if (!file.files.length) return;
    preview.src = URL.createObjectURL(file.files[0]);
    preview.style.display = 'block';
    render('<p class="small">识别中…</p>');
    const fd = new FormData();
    fd.append('file', file.files[0]);
    try {
      const resp = await fetch('/predict', { method: 'POST', body: fd });
      const d = await resp.json();
      if (d.detail) { render('<p style="color:#c33">' + d.detail + '</p>'); return; }
      let html = '<p>预测：<span class="badge">' + d.predicted_class + '</span> <b>' +
                 (d.confidence * 100).toFixed(2) + '%</b></p>';
      d.top3.forEach(t => {
        html += '<div class="small">' + t.class + ' — ' + (t.confidence * 100).toFixed(2) + '%</div>' +
                '<div class="bar"><i style="width:' + (t.confidence * 100) + '%"></i></div>';
      });
      render(html);
    } catch (e) {
      render('<p style="color:#c33">请求失败：' + e + '</p>');
    }
  }

  file.addEventListener('change', show);
  drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', e => {
    e.preventDefault(); drop.classList.remove('over');
    file.files = e.dataTransfer.files; show();
  });
</script>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok", "model": "ResNet18", "classes": len(CLASSES), "device": DEVICE}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {e}")
    top = predict_pil(img, MODEL, CLASSES, MEAN, STD, DEVICE, topk=3)
    return {"filename": file.filename, "predicted_class": top[0]["class"],
            "confidence": top[0]["confidence"], "top3": top}
