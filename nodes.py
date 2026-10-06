# -*- coding: utf-8 -*-
"""
BSAI VAE 路由节点
=================
- BSAIVAEDecodeRouter  : VAE 解码路由（CUDA / XPU(8190 worker) / CPU 按水位自动选择）
- BSAIVAEEncodeRouter  : VAE 编码路由（CUDA / CPU）

设备策略：
- cuda ：本进程 CUDA 解码（GPU1）
- xpu  ：转发 8190 XPU worker 解码（核显分担 GPU1，latent 二进制协议）
- cpu  ：本进程 CPU 解码（显存零占用，兜底）
- auto ：GPU1 显存低于阈值 → XPU（worker 在线）否则 CPU；显存充足留 CUDA

本进程 CUDA 的 torch 无 xpu 支持，故 XPU 解码必须跨进程转发到
BSAI-ComfyUI-Orchestrator 的 bsai_xpu_vae_server.py（8190）。
"""

import struct
import urllib.request

import torch

# ---- BSAI 插件协同 SDK：加载即自动注册（ComfyUI 启动自动触发） ----
import sys
import os as _os
_ORCH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                      "..", "BSAI-ComfyUI-Orchestrator")
if _ORCH not in sys.path:
    sys.path.insert(0, _ORCH)
from bsai_orch_client import BSAIOrch  # noqa: E402

BSAIOrch.register(
    name="BSAI-VAERouter",
    kind="vae_decode",                      # 能力类型：VAE 解码
    hardware=["xpu", "cuda", "cpu"],        # 偏好顺序：核显 XPU → GPU1 → CPU
    endpoint="http://127.0.0.1:8190/vae/decode",  # XPU 转发端点
    health="http://127.0.0.1:8190/health/ready",  # 健康探针
)

_VAE_DEVICE = {}  # id(vae) -> device str

_XPU_WORKER_URL = "http://127.0.0.1:8190"
_MAGIC = b"BSAIVAE1"


def _xpu_worker_online() -> bool:
    """8190 XPU worker 是否就绪（HTTP 探测）"""
    try:
        with urllib.request.urlopen(_XPU_WORKER_URL + "/health/ready", timeout=1.5) as r:
            import json
            return bool(json.loads(r.read()).get("ready", False))
    except Exception:
        return False


def _cuda_free_mb() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / (1024.0 ** 2)
    except Exception:
        pass
    return 0.0


def _pick_device(strategy: str, cuda_free_threshold_mb: int) -> str:
    """返回目标设备：cuda / xpu / cpu"""
    if strategy == "cuda":
        return "cuda"
    if strategy == "xpu":
        return "xpu" if _xpu_worker_online() else "cpu"
    if strategy == "cpu":
        return "cpu"
    # auto：GPU1 显存低于阈值 -> 分流 XPU（worker 在线）否则 CPU，否则留 CUDA
    if torch.cuda.is_available():
        free = _cuda_free_mb()
        if free < cuda_free_threshold_mb:
            return "xpu" if _xpu_worker_online() else "cpu"
        return "cuda"
    return "xpu" if _xpu_worker_online() else "cpu"


def _move_vae(vae, dev: str) -> None:
    """把 VAE 模型整体搬到目标设备（cuda/cpu，仅本进程）"""
    if dev not in ("cuda", "cpu"):
        return
    key = id(vae)
    cur = _VAE_DEVICE.get(key)
    if cur == dev:
        return
    try:
        vae.first_stage_model.to(dev)
    except Exception as exc:
        try:
            vae.first_stage_model.to("cpu")
        except Exception:
            pass
        raise RuntimeError("VAE 搬移 %s -> %s 失败: %s" % (cur, dev, exc))
    _VAE_DEVICE[key] = dev


def _xpu_worker_decode(samples, endpoint=None):
    """转发 8190 XPU worker 解码（二进制协议），返回 CPU float32 IMAGE 张量"""
    url = (endpoint or _XPU_WORKER_URL + "/vae/decode")
    lat = samples.detach().cpu().float().contiguous()
    b = lat.numpy().tobytes()
    head = _MAGIC + struct.pack("<I", lat.dim()) + struct.pack("<%dI" % lat.dim(), *lat.shape)
    req = urllib.request.Request(
        url, data=head + b,
        headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        out = resp.read()
    if len(out) < 12 or out[:8] != _MAGIC:
        raise RuntimeError("XPU worker 返回协议错误")
    ndim = struct.unpack("<I", out[8:12])[0]
    off = 12
    shape = list(struct.unpack("<%dI" % ndim, out[off:off + 4 * ndim]))
    off += 4 * ndim
    import numpy as np
    arr = np.frombuffer(out[off:], dtype=np.float32).copy()
    return torch.from_numpy(arr).reshape(shape)


def _decode_on(vae, samples, dev: str, endpoint=None):
    """在目标设备上解码（xpu 跨进程转发；cuda/cpu 本进程）"""
    if dev == "xpu":
        out = _xpu_worker_decode(samples, endpoint)
        return out.float()
    _move_vae(vae, dev)
    s = samples.to(dev)
    try:
        out = vae.first_stage_model.decode(s)
        if out is None:
            raise RuntimeError("first_stage_model.decode 返回 None")
        return out.detach().cpu().float()
    finally:
        del s


def _encode_on(vae, pixels, dev: str):
    if dev == "xpu":
        # encode 暂不支持跨进程（解码才占大头）；回退本进程 CPU 或 CUDA
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("[BSAI-VAERouter] encode 不支持 xpu 转发，回退 %s" % dev)
    _move_vae(vae, dev)
    p = pixels.to(dev)
    try:
        latent = vae.first_stage_model.encode(p)
        if isinstance(latent, (tuple, list)):
            latent = latent[0]
        return {"samples": latent.detach().cpu().float()}
    finally:
        del p


class BSAIVAEDecodeRouter:
    """VAE 解码路由：采样占 GPU1 显存大头，VAE decode 按水位分流到 XPU(8190)/CPU"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "strategy": (["auto", "cuda", "xpu", "cpu"], {"default": "auto"}),
                "cuda_free_threshold_mb": (
                    "INT",
                    {"default": 2500, "min": 0, "max": 65536, "step": 100},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "BSAI/VAE 路由"

    def decode(self, vae, latent, strategy, cuda_free_threshold_mb):
        samples = latent.get("samples")
        if samples is None:
            raise ValueError("latent 缺少 samples")
        used = None
        if strategy == "auto":
            # 自动协同：SDK 自动探活 + 水位分流 + 跨进程租约 + 端点
            alloc = BSAIOrch.allocate("vae_decode", requester="8191",
                                      gpu_free_mb=_cuda_free_mb())
            if not alloc.ok:
                # 全部离线/被占用 → 降级 CPU（显存零占用兜底）
                used = "cpu"
                print("[BSAI-VAERouter] SDK 分配失败(%s) → 降级 CPU" % alloc.reason)
                img = _decode_on(vae, samples, used)
            else:
                used = alloc.target
                try:
                    img = _decode_on(vae, samples, used, endpoint=alloc.endpoint)
                finally:
                    alloc.release()  # 停 watchdog + 释放租约
        else:
            # 显式策略：尊重用户选择（不自动协同）
            used = _pick_device(strategy, cuda_free_threshold_mb)
            img = _decode_on(vae, samples, used)
        print("[BSAI-VAERouter] decode on %s -> %s (cuda_free=%dMB)" % (
            used, tuple(img.shape), int(_cuda_free_mb())))
        return (img,)


class BSAIVAEEncodeRouter:
    """VAE 编码路由（CUDA/CPU；xpu 自动回退）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "pixels": ("IMAGE",),
                "strategy": (["auto", "cuda", "xpu", "cpu"], {"default": "auto"}),
                "cuda_free_threshold_mb": (
                    "INT",
                    {"default": 2500, "min": 0, "max": 65536, "step": 100},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "BSAI/VAE 路由"

    def encode(self, vae, pixels, strategy, cuda_free_threshold_mb):
        dev = _pick_device(strategy, cuda_free_threshold_mb)
        if dev == "xpu":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        lat = _encode_on(vae, pixels, dev)
        print("[BSAI-VAERouter] encode on %s -> %s" % (
            dev, tuple(lat["samples"].shape)))
        return (lat,)


NODE_CLASS_MAPPINGS = {
    "BSAIVAEDecodeRouter": BSAIVAEDecodeRouter,
    "BSAIVAEEncodeRouter": BSAIVAEEncodeRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIVAEDecodeRouter": "BSAI VAE 解码路由 (CUDA↔XPU8190↔CPU)",
    "BSAIVAEEncodeRouter": "BSAI VAE 编码路由 (CUDA↔CPU)",
}
