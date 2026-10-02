# -*- coding: utf-8 -*-
"""
BSAI VAE 路由节点 —— ComfyUI 插件入口
========================================
在 CUDA / Intel XPU / CPU 之间按显存水位路由 VAE 编解码：
- 采样（GPU1 显存大头）紧张时，把 VAE decode/encode 分流到核显 XPU 或 CPU，
  让 GPU1 只干采样，互不抢显存。
- latent 经 CPU 中转（张量搬运），输出统一回 CPU，兼容 ComfyUI 常规下游。
"""

import os

WEB_DIRECTORY = "./web" if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")) else None

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
