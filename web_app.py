#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI图像检测 Web服务 - Render逐张处理版
======================================
支持逐张上传检测，实时进度条，图片和结果同行显示
"""

import os
import gc
import traceback
import tempfile
import warnings
from datetime import datetime

from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

from ai_image_detector import AIImageDetector

detector = AIImageDetector()
print("AIImageDetector initialized successfully", flush=True)


HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI图像检测器</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); min-height: 100vh; color: #fff; }
        .container { max-width: 1000px; margin: 0 auto; padding: 40px 20px; }
        header { text-align: center; margin-bottom: 40px; }
        header h1 { font-size: 2.2em; background: linear-gradient(90deg, #e94560, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        header p { color: #a0a0a0; margin-top: 10px; }
        .upload-area { background: rgba(255,255,255,0.05); border: 2px dashed rgba(255,255,255,0.2); border-radius: 16px; padding: 40px 30px; text-align: center; margin-bottom: 20px; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: #e94560; background: rgba(233,69,96,0.05); }
        .upload-area h3 { font-size: 1.2em; margin-bottom: 8px; }
        .upload-area p { color: #888; font-size: 0.9em; }
        input[type="file"] { display: none; }
        .btn { display: inline-block; padding: 10px 24px; background: linear-gradient(90deg, #e94560, #ff6b6b); color: #fff; border: none; border-radius: 8px; font-size: 1em; cursor: pointer; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-secondary { background: rgba(255,255,255,0.1); margin-left: 10px; }

        /* 进度条 */
        .progress-section { display: none; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .progress-bar-bg { width: 100%; height: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; overflow: hidden; }
        .progress-bar-fill { height: 100%; background: linear-gradient(90deg, #e94560, #ff6b6b); border-radius: 10px; transition: width 0.5s ease; }
        .progress-text { text-align: center; margin-top: 10px; color: #ccc; font-size: 0.95em; }
        .progress-stats { display: flex; justify-content: center; gap: 20px; margin-top: 10px; }
        .progress-stat { text-align: center; }
        .progress-stat .num { font-size: 1.3em; font-weight: bold; }
        .progress-stat .lbl { color: #888; font-size: 0.8em; }

        /* 结果行 */
        .result-row { display: flex; align-items: flex-start; gap: 15px; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px; margin-bottom: 12px; border-left: 3px solid #555; }
        .result-row.ai { border-left-color: #e74c3c; }
        .result-row.real { border-left-color: #2ecc71; }
        .result-row.suspicious { border-left-color: #f39c12; }
        .result-row.error { border-left-color: #888; }

        .result-img { width: 120px; height: 120px; border-radius: 8px; object-fit: cover; flex-shrink: 0; }
        .result-content { flex: 1; min-width: 0; }
        .result-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
        .result-filename { font-weight: bold; font-size: 0.9em; color: #ddd; }
        .status-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-weight: bold; font-size: 0.75em; }
        .CLEAR { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
        .SUSPICIOUS { background: rgba(243,156,18,0.2); color: #f39c12; border: 1px solid rgba(243,156,18,0.4); }
        .CRITICAL { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }
        .result-verdict { font-weight: bold; font-size: 0.9em; }
        .result-metrics { display: flex; gap: 15px; margin-bottom: 10px; }
        .result-metric { font-size: 0.85em; }
        .result-metric .val { font-weight: bold; }
        .indicators { display: flex; flex-direction: column; gap: 4px; }
        .indicator { display: flex; align-items: center; }
        .indicator .name { width: 70px; font-size: 0.75em; color: #aaa; }
        .indicator .bar { flex: 1; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px; overflow: hidden; margin: 0 6px; }
        .indicator .fill { height: 100%; border-radius: 5px; transition: width 1s; }
        .indicator .score { width: 35px; text-align: right; font-size: 0.75em; }

        .error-msg { color: #e74c3c; font-size: 0.85em; }
        .pending { color: #888; font-size: 0.85em; font-style: italic; }

        footer { text-align: center; padding: 30px; color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI图像检测器</h1>
            <p>上传图片，逐张检测是否为AI生成</p>
        </header>
        <div class="upload-area" id="uploadArea">
            <h3>点击上传图片</h3>
            <p>支持 JPG、PNG、WebP 格式，最多5张，每张最大16MB</p>
            <input type="file" id="fileInput" accept="image/*" multiple>
            <button class="btn" id="selectBtn">选择图片</button>
        </div>
        <div class="progress-section" id="progressSection">
            <div class="progress-bar-bg"><div class="progress-bar-fill" id="progressFill" style="width:0%"></div></div>
            <div class="progress-text" id="progressText">准备检测...</div>
            <div class="progress-stats">
                <div class="progress-stat"><div class="num" id="statTotal">0</div><div class="lbl">总数</div></div>
                <div class="progress-stat"><div class="num" id="statDone" style="color:#2ecc71">0</div><div class="lbl">已完成</div></div>
                <div class="progress-stat"><div class="num" id="statAI" style="color:#e74c3c">0</div><div class="lbl">AI生成</div></div>
            </div>
        </div>
        <div id="results"></div>
        <footer>
            <p>AI图像检测器 | 六维度分析 | 结果仅供参考</p>
        </footer>
    </div>
    <script>
        const uploadArea = document.getElementById("uploadArea");
        const fileInput = document.getElementById("fileInput");
        const selectBtn = document.getElementById("selectBtn");
        const progressSection = document.getElementById("progressSection");
        const progressFill = document.getElementById("progressFill");
        const progressText = document.getElementById("progressText");
        const statTotal = document.getElementById("statTotal");
        const statDone = document.getElementById("statDone");
        const statAI = document.getElementById("statAI");
        const resultsDiv = document.getElementById("results");

        selectBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
        uploadArea.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", async () => {
            const files = Array.from(fileInput.files).slice(0, 5);
            if (!files.length) return;

            fileInput.value = "";
            progressSection.style.display = "block";
            statTotal.textContent = files.length;
            statDone.textContent = 0;
            statAI.textContent = 0;
            resultsDiv.innerHTML = "";

            // 创建结果行（初始状态）
            const resultRows = [];
            files.forEach((file, i) => {
                const row = document.createElement("div");
                row.className = "result-row";
                row.id = "row-" + i;

                const reader = new FileReader();
                reader.onload = (e) => {
                    row.innerHTML = '<img class="result-img" src="' + e.target.result + '"><div class="result-content"><div class="pending">等待检测...</div></div>';
                };
                reader.readAsDataURL(file);

                resultsDiv.appendChild(row);
                resultRows.push({ row, file });
            });

            // 逐张检测
            let doneCount = 0;
            let aiCount = 0;

            for (let i = 0; i < resultRows.length; i++) {
                const { row, file } = resultRows[i];
                progressText.textContent = "正在检测第 " + (i + 1) + "/" + files.length + " 张...";
                progressFill.style.width = (i / files.length * 100) + "%";

                const formData = new FormData();
                formData.append("image", file);

                try {
                    const res = await fetch("/api/detect", { method: "POST", body: formData });
                    const text = await res.text();
                    if (!res.ok) {
                        row.className = "result-row error";
                        row.innerHTML = '<img class="result-img" src="' + row.querySelector("img").src + '"><div class="result-content"><div class="error-msg">检测失败: HTTP ' + res.status + '</div></div>';
                        continue;
                    }
                    const data = JSON.parse(text);
                    if (!data.success) {
                        row.className = "result-row error";
                        row.innerHTML = '<img class="result-img" src="' + row.querySelector("img").src + '"><div class="result-content"><div class="error-msg">' + data.error + '</div></div>';
                        continue;
                    }

                    // 成功
                    const cardClass = data.is_ai_generated ? 'ai' : (data.status === 'SUSPICIOUS' ? 'suspicious' : 'real');
                    const verdictColor = data.is_ai_generated ? '#e74c3c' : '#2ecc71';
                    const verdictText = data.is_ai_generated ? 'AI生成' : '真实照片';
                    row.className = "result-row " + cardClass;

                    let html = '<div class="result-header">';
                    html += '<span class="result-filename">' + data.filename + '</span>';
                    html += '<span class="status-badge ' + data.status + '">' + data.status + '</span>';
                    html += '<span class="result-verdict" style="color:' + verdictColor + '">' + verdictText + '</span>';
                    html += '</div>';

                    html += '<div class="result-metrics">';
                    html += '<div class="result-metric">AI概率: <span class="val">' + (data.ai_probability*100).toFixed(1) + '%</span></div>';
                    html += '<div class="result-metric">置信度: <span class="val">' + data.confidence + '%</span></div>';
                    html += '<div class="result-metric">信任分: <span class="val">' + data.trust_score + '</span></div>';
                    html += '</div>';

                    html += '<div class="indicators">';
                    for (const [name, score] of Object.entries(data.indicators)) {
                        const color = score > 60 ? '#e74c3c' : (score > 35 ? '#f39c12' : '#2ecc71');
                        html += '<div class="indicator"><span class="name">' + name + '</span><div class="bar"><div class="fill" style="width:' + score + '%;background:' + color + '"></div></div><span class="score">' + score.toFixed(1) + '%</span></div>';
                    }
                    html += '</div>';

                    row.querySelector(".result-content").innerHTML = html;

                    doneCount++;
                    if (data.is_ai_generated) aiCount++;
                    statDone.textContent = doneCount;
                    statAI.textContent = aiCount;

                } catch (e) {
                    row.className = "result-row error";
                    row.innerHTML = '<img class="result-img" src="' + row.querySelector("img").src + '"><div class="result-content"><div class="error-msg">网络错误: ' + e.message + '</div></div>';
                }
            }

            progressFill.style.width = "100%";
            progressText.textContent = "检测完成！";
        });
    </script>
</body>
</html>'''


@app.errorhandler(Exception)
def handle_error(e):
    return f"<h1>Error</h1><pre>{str(e)}\n\n{traceback.format_exc()}</pre>", 500


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/api/detect', methods=['POST'])
def api_detect():
    if 'image' not in request.files:
        return jsonify({'error': '未上传图像'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        result = detector.detect(filepath, max_size=512)

        response = {
            'success': True,
            'filename': result.filename,
            'status': result.status,
            'is_ai_generated': bool(result.is_ai_generated),
            'ai_probability': float(result.ai_probability),
            'confidence': float(result.confidence),
            'trust_score': float(result.trust_score),
            'indicators': {k: float(v) for k, v in result.indicators.items()},
            'timestamp': timestamp
        }
        return jsonify(response)
    except Exception as e:
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        print(error_detail, flush=True)
        return jsonify({'error': error_detail}), 500
    finally:
        try:
            os.unlink(filepath)
        except Exception:
            pass
        gc.collect()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
