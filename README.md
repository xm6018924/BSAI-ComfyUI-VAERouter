# BSAI-ComfyUI-VAERouter

VAE 解码/编码智能路由节点 —— 按 GPU1 显存水位自动选择解码设备：CUDA（GPU1）/ XPU（8190 核显 worker）/ CPU（兜底），把 VAE 解码负载从独显分流到 Intel 核显，释放独显显存给采样。

Smart VAE decode/encode router — automatically picks decode device by GPU1 VRAM headroom: CUDA (GPU1) / XPU (8190 iGPU worker) / CPU (fallback). Offloads VAE decode from the discrete GPU to Intel iGPU, freeing VRAM for sampling.

---

## Nodes | 节点

| Node | 说明 |
|---|---|
| **BSAIVAEDecodeRouter** | VAE 解码路由：auto / cuda / xpu / cpu |
| **BSAIVAEEncodeRouter** | VAE 编码路由：cuda / cpu |

## Device Strategy | 设备策略

| Mode | Behavior |
|---|---|
| **cuda** | Decode on GPU1 (default, fastest) |
| **xpu** | Forward to XPU worker at `http://127.0.0.1:8190` (latent binary protocol) |
| **cpu** | Decode on CPU, zero VRAM (slow fallback) |
| **auto** | If GPU1 free VRAM below threshold AND XPU worker online → xpu; if below threshold but worker offline → cpu; else cuda |

## Prerequisite | 前置

The XPU worker (port 8190) must be running. It is launched automatically by the BSAI launcher bat (`BSAI_XPU_VAE_Worker_8190.bat`), which starts `BSAI_XPU_VAE_Server.py`.

## Install | 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-VAERouter.git
```

No extra pip dependencies beyond ComfyUI base.

## Usage | 使用

1. Start the XPU VAE worker (auto-started by BSAI launcher).
2. Replace your regular `VAEDecode` node with **BSAIVAEDecodeRouter**.
3. Set mode to `auto` for hands-off offloading.
4. Connect model/latent/images exactly like the standard VAE node.

## Compatibility | 兼容

- ComfyUI ≥ 0.37
- Windows 10/11 with NVIDIA dGPU + Intel iGPU
- Pairs with **BSAI-ComfyUI-Orchestrator** for policy-driven routing.

## License | 许可

MIT
