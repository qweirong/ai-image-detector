#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI图像检测 Web服务 - Render最简稳定版（支持批量检测）
=====================================================
仅依赖 numpy + Pillow + Flask，确保 Render 稳定运行
支持同时上传最多10张图片进行批量检测
"""

import os
import traceback
import tempfile
import warnings
from datetime import datetime

from flask import Flask, request, jsonify

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB（支持多张）

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
        .upload-area { background: rgba(255,255,255,0.05); border: 2px dashed rgba(255,255,255,0.2); border-radius: 16px; padding: 50px 30px; text-align: center; margin-bottom: 20px; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: #e94560; background: rgba(233,69,96,0.05); }
        .upload-area h3 { font-size: 1.2em; margin-bottom: 8px; }
        .upload-area p { color: #888; font-size: 0.9em; }
        input[type="file"] { display: none; }
        .btn { display: inline-block; padding: 12px 28px; background: linear-gradient(90deg, #e94560, #ff6b6b); color: #fff; border: none; border-radius: 8px; font-size: 1em; cursor: pointer; margin-top: 15px; }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-secondary { background: rgba(255,255,255,0.1); margin-left: 10px; }
        .preview-section { display: none; margin-bottom: 20px; }
        .preview-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
        .preview-item { position: relative; width: 100px; height: 100px; border-radius: 10px; overflow: hidden; background: rgba(255,255,255,0.05); }
        .preview-item img { width: 100%; height: 100%; object-fit: cover; }
        .preview-item .remove { position: absolute; top: 2px; right: 2px; background: rgba(231,76,60,0.9); color: #fff; border: none; border-radius: 50%; width: 20px; height: 20px; cursor: pointer; font-size: 12px; line-height: 20px; text-align: center; }
        .preview-info { color: #888; font-size: 0.85em; }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner { width: 40px; height: 40px; border: 3px solid rgba(255,255,255,0.1); border-top-color: #e94560; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result { display: none; background: rgba(255,255,255,0.05); border-radius: 16px; padding: 25px; margin-bottom: 20px; }
        .status-badge { display: inline-block; padding: 6px 16px; border-radius: 16px; font-weight: bold; font-size: 0.9em; }
        .CLEAR { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
        .SUSPICIOUS { background: rgba(243,156,18,0.2); color: #f39c12; border: 1px solid rgba(243,156,18,0.4); }
        .CRITICAL { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }
        .batch-summary { display: flex; gap: 15px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }
        .batch-card { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 15px 25px; text-align: center; }
        .batch-card .val { font-size: 2em; font-weight: bold; }
        .batch-card .lbl { color: #888; font-size: 0.85em; }
        .result-card { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 15px; margin-bottom: 12px; border-left: 3px solid #555; }
        .result-card.ai { border-left-color: #e74c3c; }
        .result-card.real { border-left-color: #2ecc71; }
        .result-card.suspicious { border-left-color: #f39c12; }
        .result-card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .result-card-header .filename { font-weight: bold; font-size: 0.95em; }
        .result-card-header .verdict { font-weight: bold; font-size: 0.9em; }
        .indicator { display: flex; align-items: center; margin: 6px 0; }
        .indicator .name { width: 80px; font-size: 0.8em; color: #aaa; }
        .indicator .bar { flex: 1; height: 12px; background: rgba(255,255,255,0.1); border-radius: 6px; overflow: hidden; margin: 0 8px; }
        .indicator .fill { height: 100%; border-radius: 6px; transition: width 1s; }
        .indicator .score { width: 40px; text-align: right; font-size: 0.8em; }
        footer { text-align: center; padding: 30px; color: #666; font-size: 0.85em; }
        .error-box { background: rgba(231,76,60,0.1); border: 1px solid rgba(231,76,60,0.3); border-radius: 10px; padding: 15px; margin: 15px 0; color: #e74c3c; font-size: 0.9em; word-break: break-all; }
        .progress { display: none; text-align: center; margin-bottom: 10px; }
        .progress-bar { width: 100%; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #e94560, #ff6b6b); border-radius: 3px; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI图像检测器</h1>
            <p>上传图片，检测是否为AI生成（支持同时上传最多10张）</p>
        </header>
        <div class="upload-area" id="uploadArea">
            <h3>点击上传图片</h3>
            <p>支持 JPG、PNG、WebP 格式，最多10张，每张最大16MB</p>
            <input type="file" id="fileInput" accept="image/*" multiple>
            <button class="btn" id="selectBtn">选择图片</button>
        </div>
        <div class="preview-section" id="previewSection">
            <div class="preview-grid" id="previewGrid"></div>
            <div class="preview-info" id="previewInfo"></div>
            <button class="btn" id="detectBtn">开始检测</button>
            <button class="btn btn-secondary" id="clearBtn">清空</button>
        </div>
        <div class="progress" id="progress">
            <div class="progress-bar"><div class="progress-fill" id="progressFill" style="width:0%"></div></div>
            <p id="progressText" style="margin-top:8px;color:#888;font-size:0.85em;">正在分析...</p>
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
        const detectBtn = document.getElementById("detectBtn");
        const clearBtn = document.getElementById("clearBtn");
        const previewSection = document.getElementById("previewSection");
        const previewGrid = document.getElementById("previewGrid");
        const previewInfo = document.getElementById("previewInfo");
        const progress = document.getElementById("progress");
        const progressFill = document.getElementById("progressFill");
        const progressText = document.getElementById("progressText");
        const result = document.getElementById("result");

        let selectedFiles = [];

        selectBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
        uploadArea.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", () => {
            const newFiles = Array.from(fileInput.files);
            if (selectedFiles.length + newFiles.length > 10) {
                newFiles = newFiles.slice(0, 10 - selectedFiles.length);
            }
            selectedFiles = selectedFiles.concat(newFiles);
            renderPreviews();
        });

        clearBtn.addEventListener("click", () => {
            selectedFiles = [];
            fileInput.value = "";
            renderPreviews();
            result.style.display = "none";
        });

        function renderPreviews() {
            previewGrid.innerHTML = "";
            if (selectedFiles.length === 0) {
                previewSection.style.display = "none";
                return;
            }
            previewSection.style.display = "block";
            previewInfo.textContent = "已选择 " + selectedFiles.length + "/10 张图片";
            selectedFiles.forEach((file, i) => {
                const div = document.createElement("div");
                div.className = "preview-item";
                const reader = new FileReader();
                reader.onload = (e) => {
                    div.innerHTML = '<img src="' + e.target.result + '"><button class="remove" onclick="removeFile(' + i + ')">×</button>';
                };
                reader.readAsDataURL(file);
                previewGrid.appendChild(div);
            });
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            renderPreviews();
        }

        detectBtn.addEventListener("click", async () => {
            if (!selectedFiles.length) return;
            progress.style.display = "block";
            result.style.display = "none";
            detectBtn.disabled = true;

            const formData = new FormData();
            selectedFiles.forEach(f => formData.append("images", f));

            try {
                const res = await fetch("/api/detect_batch", { method: "POST", body: formData });
                const text = await res.text();
                if (!res.ok) {
                    result.innerHTML = '<div class="error-box"><b>服务器错误 (HTTP ' + res.status + ')</b><br><pre style="white-space:pre-wrap;font-size:12px;margin-top:8px;">' + text.replace(/</g, "&lt;") + '</pre></div>';
                    result.style.display = "block";
                    return;
                }
                const data = JSON.parse(text);
                if (!data.success) throw new Error(data.error);

                let html = '<div class="batch-summary">';
                html += '<div class="batch-card"><div class="val">' + data.total + '</div><div class="lbl">总图片数</div></div>';
                html += '<div class="batch-card"><div class="val" style="color:#e74c3c">' + data.ai_count + '</div><div class="lbl">AI生成</div></div>';
                html += '<div class="batch-card"><div class="val" style="color:#2ecc71">' + (data.total - data.ai_count) + '</div><div class="lbl">真实照片</div></div>';
                html += '<div class="batch-card"><div class="val" style="color:#f39c12">' + data.suspicious_count + '</div><div class="lbl">可疑</div></div>';
                html += '</div>';
                html += '<div style="text-align:center;color:#666;font-size:0.85em;margin-bottom:15px;">检测时间：' + data.timestamp + '</div>';

                data.results.forEach(r => {
                    const cardClass = r.is_ai_generated ? 'ai' : (r.status === 'SUSPICIOUS' ? 'suspicious' : 'real');
                    const verdictColor = r.is_ai_generated ? '#e74c3c' : '#2ecc71';
                    const verdictText = r.is_ai_generated ? 'AI生成' : '真实照片';

                    html += '<div class="result-card ' + cardClass + '">';
                    html += '<div class="result-card-header">';
                    html += '<span class="filename">' + r.filename + '</span>';
                    html += '<span class="status-badge ' + r.status + '">' + r.status + '</span>';
                    html += '<span class="verdict" style="color:' + verdictColor + '">' + verdictText + ' (' + (r.ai_probability * 100).toFixed(1) + '%)</span>';
                    html += '</div>';

                    for (const [name, score] of Object.entries(r.indicators)) {
                        const color = score > 60 ? '#e74c3c' : (score > 35 ? '#f39c12' : '#2ecc71');
                        html += '<div class="indicator"><span class="name">' + name + '</span><div class="bar"><div class="fill" style="width:' + score + '%;background:' + color + '"></div></div><span class="score">' + score.toFixed(1) + '%</span></div>';
                    }
                    html += '</div>';
                });

                result.innerHTML = html;
                result.style.display = "block";
            } catch (e) {
                result.innerHTML = '<div class="error-box">错误: ' + e.message + '</div>';
                result.style.display = "block";
            } finally {
                progress.style.display = "none";
                detectBtn.disabled = false;
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


@app.route('/api/detect_batch', methods=['POST'])
def api_detect_batch():
    if 'images' not in request.files:
        return jsonify({'error': '未上传图像'}), 400

    files = request.files.getlist('images')

    # 限制最多10张
    files = files[:10]

    results = []

    for file in files:
        if file.filename == '':
            continue

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = detector.detect(filepath, max_size=1024)

            results.append({
                'filename': result.filename,
                'status': result.status,
                'is_ai_generated': bool(result.is_ai_generated),
                'ai_probability': float(result.ai_probability),
                'confidence': float(result.confidence),
                'trust_score': float(result.trust_score),
                'indicators': {k: float(v) for k, v in result.indicators.items()}
            })
        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'ERROR',
                'is_ai_generated': False,
                'ai_probability': 0.0,
                'confidence': 0.0,
                'trust_score': 0.0,
                'indicators': {},
                'error': str(e)
            })
        finally:
            try:
                os.unlink(filepath)
            except Exception:
                pass

    ai_count = sum(1 for r in results if r.get('is_ai_generated'))
    suspicious_count = sum(1 for r in results if r.get('status') == 'SUSPICIOUS')

    return jsonify({
        'success': True,
        'total': len(results),
        'ai_count': ai_count,
        'suspicious_count': suspicious_count,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
