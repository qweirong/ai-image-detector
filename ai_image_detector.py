#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI图像检测程序 - 最简稳定版（Render专用）
==========================================
仅依赖 numpy + Pillow，无外部复杂库，确保 Render 稳定运行
"""

import os
import io
import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

warnings.filterwarnings('ignore')


@dataclass
class DetectionResult:
    filename: str
    is_ai_generated: bool
    confidence: float
    ai_probability: float
    trust_score: float
    indicators: Dict[str, float]
    status: str


class AIImageDetector:
    """AI图像检测器 - 最简版"""

    def __init__(self):
        pass

    def detect(self, image_path: str, max_size: int = 1024) -> DetectionResult:
        image = self._load_image(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 缩放大图，防止内存不足
        h, w = image.shape[:2]
        if max(h, w) > max_size:
            pil_img = Image.fromarray(image)
            pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
            image = np.array(pil_img)

        filename = os.path.basename(image_path)
        h, w = image.shape[:2]

        # 1. 频域分析 - 修复：真实照片高频多，不应直接乘3
        gray = np.mean(image, axis=2).astype(np.uint8) if len(image.shape) == 3 else image
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        mask_low = (x - cx)**2 + (y - cy)**2 <= radius**2
        high_freq = np.sum(magnitude[~mask_low]) / (np.sum(magnitude) + 1e-10)
        # 真实照片高频比例通常在 0.3-0.5，AI图像可能更极端
        # 用高斯函数将 0.3-0.7 映射到 0-1
        freq_score = max(0, min(abs(high_freq - 0.5) * 3, 1.0))

        # 2. ELA噪声分析
        try:
            pil_img = Image.fromarray(image.astype(np.uint8))
            buf = io.BytesIO()
            pil_img.save(buf, 'JPEG', quality=90)
            buf.seek(0)
            recompressed = np.array(Image.open(buf))
            diff = np.abs(image.astype(float) - recompressed.astype(float))
            ela_score = min(np.std(diff / 255.0) * 5, 1.0)
        except:
            ela_score = 0.5

        # 3. 颜色平滑度 - 修复：真实照片也有平滑区域
        grad_x = np.abs(np.diff(image.astype(float), axis=1))
        grad_y = np.abs(np.diff(image.astype(float), axis=0))
        all_grads = np.concatenate([grad_x.flatten(), grad_y.flatten()])
        # 使用梯度分布的变异系数（纯numpy）
        grad_mean = np.mean(all_grads)
        grad_std = np.std(all_grads)
        cv = grad_std / (grad_mean + 1e-10) if grad_mean > 0 else 0
        # 真实照片梯度变化更自然（cv适中），AI可能过于均匀
        color_score = max(0, min(abs(cv - 2.0) / 2.0, 1.0))

        # 4. 纹理LBP
        lbp_score = self._lbp_score(gray)

        # 5. 元数据
        meta_score = self._metadata_score(image_path, image)

        # 6. 边缘分析 - 修复：真实照片边缘密度也高
        gx = np.abs(np.diff(gray.astype(float), axis=1))
        gy = np.abs(np.diff(gray.astype(float), axis=0))
        edge_mag = np.sqrt(gx[:-1,:]**2 + gy[:,:-1]**2)
        # 计算边缘的均匀性而非密度
        edge_mean = np.mean(edge_mag)
        edge_std = np.std(edge_mag)
        cv = edge_std / (edge_mean + 1e-10) if edge_mean > 0 else 0
        # 真实照片边缘变化更自然（cv适中），AI可能过于均匀或过于杂乱
        edge_score = max(0, min(abs(cv - 1.5) / 1.5, 1.0))

        # 加权计算
        weights = {'freq': 0.20, 'ela': 0.20, 'color': 0.15, 'lbp': 0.15, 'meta': 0.15, 'edge': 0.15}
        ai_probability = (
            freq_score * weights['freq'] +
            ela_score * weights['ela'] +
            color_score * weights['color'] +
            lbp_score * weights['lbp'] +
            meta_score * weights['meta'] +
            edge_score * weights['edge']
        )

        # 判断状态
        if ai_probability < 0.35:
            status = 'CLEAR'
            is_ai = False
        elif ai_probability < 0.65:
            status = 'SUSPICIOUS'
            is_ai = ai_probability > 0.5
        else:
            status = 'CRITICAL'
            is_ai = True

        indicators = {
            '频域分析': round(freq_score * 100, 1),
            '噪声分析': round(ela_score * 100, 1),
            '颜色分析': round(color_score * 100, 1),
            '纹理分析': round(lbp_score * 100, 1),
            '元数据分析': round(meta_score * 100, 1),
            '边缘分析': round(edge_score * 100, 1)
        }

        return DetectionResult(
            filename=filename,
            is_ai_generated=is_ai,
            confidence=70.0,
            ai_probability=round(ai_probability, 4),
            trust_score=round((1 - ai_probability) * 100, 1),
            indicators=indicators,
            status=status
        )

    def _load_image(self, path: str) -> Optional[np.ndarray]:
        try:
            return np.array(Image.open(path).convert('RGB'))
        except Exception as e:
            print(f"加载失败: {e}")
            return None

    def _lbp_score(self, gray):
        h, w = gray.shape
        if h < 3 or w < 3:
            return 0.5
        center = gray[1:-1, 1:-1]
        neighbors = [gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
                     gray[1:-1, 2:], gray[2:, 2:], gray[2:, 1:-1],
                     gray[2:, 0:-2], gray[1:-1, 0:-2]]
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        for i, n in enumerate(neighbors):
            lbp += ((n >= center) << i).astype(np.uint8)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)
        return min(np.sum(hist ** 2) * 10, 1.0)

    def _metadata_score(self, image_path, image):
        try:
            exif = Image.open(image_path)._getexif()
        except:
            exif = None

        has_exif = 1.0 if exif and len(exif) > 0 else 0.0
        has_camera = 0.0
        if exif:
            for tid, val in exif.items():
                if TAGS.get(tid, tid) in ['Make', 'Model', 'FNumber']:
                    has_camera = 1.0

        h, w = image.shape[:2]
        ai_sizes = [(512,512),(768,768),(1024,1024),(512,768),(768,512)]
        dim_score = 0.3 if (w,h) in ai_sizes or (h,w) in ai_sizes else 0.0

        completeness = has_exif * 0.3 + has_camera * 0.3 + (1 - dim_score) * 0.4
        return 1 - completeness
