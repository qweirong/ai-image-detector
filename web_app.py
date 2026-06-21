#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI图像检测 Web服务 - Render最简稳定版
=====================================
仅依赖 numpy + Pillow + Flask，确保 Render 稳定运行
"""

import os
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

# 导入检测器
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
        .container { max-width: 900px; margin: 0 auto; padding: 40px 20px; }
        header { text-align: center; margin-bottom: 40px; }
        header h1 { font-size: 2.2em; background: linear-gradient(90deg, #e94560, #ff6b6b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        header p { color: #a0a0a0; margin-top: 10px; }
        .upload-area { background: rgba(255,255,255,0.05); border: 2px dashed rgba(255,255,255,0.2); border-radius: 16px; padding: 50px 30px; text-align: center; margin-bottom: 30px; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: #e94560; background: rgba(233,69,96,0.05); }
        .upload-area h3 { font-size: 1.2em; margin-bottom: 8px; }
        .upload-area p { color: #888; font-size: 0.9em; }
        input[type="file"] { display: none; }
        .btn { display: inline-block; padding: 12px 28px; background: linear-gradient(90deg, #e94560, #ff6b6b); color: #fff; border: none; border-radius: 8px; font-size: 1em; cursor: pointer; margin-top: 15px; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner { width: 40px; height: 40px; border: 3px solid rgba(255,255,255,0.1); border-top-color: #e94560; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result { display: none; background: rgba(255,255,255,0.05); border-radius: 16px; padding: 25px; margin-bottom: 20px; }
        .status-badge { display: inline-block; padding: 6px 16px; border-radius: 16px; font-weight: bold; }
        .CLEAR { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
        .SUSPICIOUS { background: rgba(243,156,18,0.2); color: #f39c12; border: 1px solid rgba(243,156,18,0.4); }
        .CRITICAL { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }
        .metric { display: inline-block; background: rgba(255,255,255,0.05); border-radius: 10px; padding: 15px 25px; margin: 8px; text-align: center; }
        .metric .val { font-size: 1.8em; font-weight: bold; }
        .metric .lbl { color: #888; font-size: 0.85em; }
        .indicator { display: flex; align-items: center; margin: 10px 0; }
        .indicator .name { width: 90px; font-size: 0.85em; color: #aaa; }
        .indicator .bar { flex: 1; height: 16px; background: rgba(255,255,255,0.1); border-radius: 8px; overflow: hidden; margin: 0 10px; }
        .indicator .fill { height: 100%; border-radius: 8px; transition: width 1s; }
        .indicator .score { width: 45px; text-align: right; font-size: 0.85em; }
        footer { text-align: center; padding: 30px; color: #666; font-size: 0.85em; }
        .error-box { background: rgba(231,76,60,0.1); border: 1px solid rgba(231,76,60,0.3); border-radius: 10px; padding: 15px; margin: 15px 0; color: #e74c3c; font-size: 0.9em; word-break: break-all; }
        #imagePreview { text-align: center; margin-bottom: 20px; }
        #imagePreview img { max-width: 100%; max-height: 300px; border-radius: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI图像检测器</h1>
            <p>上传图片，检测是否为AI生成</p>
        </header>
        <div class="upload-area" id="uploadArea">
            <h3>点击上传图片</h3>
            <p>支持 JPG、PNG、WebP 格式，最大 16MB</p>
            <input type="file" id="fileInput" accept="image/*">
            <button class="btn" id="selectBtn">选择图片</button>
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在分析...</p>
        </div>
        <div class="result" id="result"></div>
        <footer>
            <p>AI图像检测器 | 六维度分析 | 结果仅供参考</p>
        </footer>
    </div>
    <script>
        const uploadArea = document.getElementById("uploadArea");
        const fileInput = document.getElementById("fileInput");
        const selectBtn = document.getElementById("selectBtn");
        const loading = document.getElementById("loading");
        const result = document.getElementById("result");

        selectBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
        uploadArea.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", async () => {
            if (!fileInput.files.length) return;
            loading.style.display = "block";
            result.style.display = "none";

            const formData = new FormData();
            formData.append("image", fileInput.files[0]);

            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.createElement('div');
                preview.id = 'imagePreview';
                preview.innerHTML = '<img src="' + e.target.result + '">';
                const existing = document.getElementById('imagePreview');
                if (existing) existing.remove();
                document.querySelector('.upload-area').after(preview);
            };
            reader.readAsDataURL(fileInput.files[0]);

            try {
                const res = await fetch("/api/detect", { method: "POST", body: formData });
                const text = await res.text();
                if (!res.ok) {
                    result.innerHTML = '<div class="error-box"><b>服务器错误 (HTTP ' + res.status + ')</b><br><pre style="white-space:pre-wrap;font-size:12px;margin-top:8px;">' + text.replace(/</g, "&lt;") + '</pre></div>';
                    result.style.display = "block";
                    return;
                }
                if (!text) {
                    result.innerHTML = '<div class="error-box">服务器返回空响应，请稍后重试</div>';
                    result.style.display = "block";
                    return;
                }
                const data = JSON.parse(text);
                if (!data.success) throw new Error(data.error);

                let html = '<div style="text-align:center;margin-bottom:20px;">';
                html += '<span class="status-badge ' + data.status + '">' + data.status + '</span>';
                html += '<h2 style="margin-top:15px;color:' + (data.is_ai_generated?'#e74c3c':'#2ecc71') + '">' + (data.is_ai_generated?'AI生成图像':'真实照片') + '</h2>';
                html += '</div>';

                html += '<div style="text-align:center;margin-bottom:20px;">';
                html += '<div class="metric"><div class="val">' + (data.ai_probability*100).toFixed(1) + '%</div><div class="lbl">AI概率</div></div>';
                html += '<div class="metric"><div class="val">' + data.confidence + '%</div><div class="lbl">置信度</div></div>';
                html += '<div class="metric"><div class="val">' + data.trust_score + '</div><div class="lbl">信任分</div></div>';
                html += '</div>';

                html += '<h4 style="margin-bottom:12px;color:#ccc;">各维度AI指标</h4>';
                for (const [name, score] of Object.entries(data.indicators)) {
                    const color = score > 60 ? '#e74c3c' : (score > 35 ? '#f39c12' : '#2ecc71');
                    html += '<div class="indicator"><span class="name">' + name + '</span><div class="bar"><div class="fill" style="width:' + score + '%;background:' + color + '"></div></div><span class="score">' + score.toFixed(1) + '%</span></div>';
                }
                result.innerHTML = html;
                result.style.display = "block";
            } catch (e) {
                result.innerHTML = '<div class="error-box">错误: ' + e.message + '</div>';
                result.style.display = "block";
            } finally {
                loading.style.display = "none";
            }
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
        # 限制图片大小，防止内存不足
        result = detector.detect(filepath, max_size=1024)

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
