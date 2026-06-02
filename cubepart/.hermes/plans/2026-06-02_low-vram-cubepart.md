# Plan: VRAM Optimization for CubePart Gradio Demo

## Goal

Reduce peak VRAM usage of the CubePart Gradio demo so it can run on an RTX 3090 (24 GB).
Currently all three large models are loaded to GPU at startup and stay resident.

## Current Architecture & VRAM Breakdown

The `ShapeDenoiserPipeline.__init__()` (in `cube_part/pipelines/shape_denoiser.py:72`)
instantiates `ShapeDenoiserSystem(config.system).eval().to(device)`, which puts **all**
sub-models on GPU at once. The system contains three independent components:

| Component | Class | Est. VRAM (bf16/fp16) |
|-----------|-------|----------------------|
| 1. Text Encoder (Qwen3-VL-4B-Instruct) | `Qwen3VLProcessor` in `condition_processors.py` | ~8 GB |
| 2. Shape VAE (encoder + decoder + occupancy) | `OneDGridAutoEncoder` in `one_d_grid_autoencoder.py` | ~2-3 GB |
| 3. Diffusion Transformer (Qwen-Image DiT, 23 layers) | `QwenImageTransformer2DModel` in `hijack_qwenimage_multi.py` | ~6-8 GB |
| **Total** | | **~16-19 GB (models only)** |

Plus activation tensors during inference (especially diffusion loop with CFG doubling),
this exceeds 24 GB.

## Inference Flow (what uses what)

From `PartShapeDenoiserPipeline.input_to_part_shape()` in `shape_denoiser.py`:

1. **Encode phase**: `pipe.encode_shape(surface)` — uses **VAE encoder** only
2. **Text encode**: `self.system.base_model(prompts)` — uses **text encoder** only
3. **Diffusion loop**: 50-step denoising — uses **diffusion transformer** only (text encoder result cached as tensors)
4. **Decode phase**: `pipe.decode_shape(latents)` + `extract_geometry()` — uses **VAE decoder + occupancy decoder** only

Key insight: the three components are **never needed simultaneously**. Each phase is
self-contained and produces output tensors that the next phase consumes.

## Proposed Approach: Phased Load/Unload

Wrap each of the three models behind a lazy-load helper that:
- Loads the model to GPU only when first accessed
- Moves it back to CPU (and `torch.cuda.empty_cache()`) when explicitly released
- Can be triggered at the Gradio runner level

### Phase breakdown per inference request:

```
Request starts
  -> Load VAE encoder to GPU
  -> encode_shape(surface)        [VAE encoder only]
  -> Unload VAE encoder to CPU
  -> Load text encoder to GPU
  -> base_model(prompts)          [text encoder only]
  -> Unload text encoder to CPU
  -> Load diffusion transformer to GPU
  -> 50-step diffusion loop      [DiT only]
  -> Unload diffusion transformer to CPU
  -> Load VAE decoder + occupancy decoder to GPU
  -> decode_shape + extract_geometry  [VAE decoder only]
  -> Unload VAE decoder to CPU
Request done
```

Peak VRAM = max(single component) + activation tensors ≈ 8-10 GB instead of 16-19 GB.

### VAE split consideration

The `OneDGridAutoEncoder` contains both encoder and decoder. We could:
- Option A: Split the VAE into encoder/decoder sub-modules and load/unload them separately (more savings)
- Option B: Keep the VAE as one unit, load it for encode, unload, then reload for decode (simpler, slightly less optimal since decoder + occupancy are smaller than encoder)

**Recommended: Option A** — the encoder is the heaviest part (9 encoder layers with cross-attention on up to 128K surface points). The decoder (16 layers, but much shorter sequences) + occupancy decoder are lighter. Splitting gives better peak VRAM.

Actually, on second thought: the occupancy decoder is queried against potentially millions of grid points (marching cubes), and the decoder itself processes all latents. The decoder+occupancy may be comparable. Let's keep it as one VAE unit for simplicity (Option B) — the savings from splitting VAE encoder vs decoder are marginal compared to the big win of not having text encoder or DiT resident.

**Revised: Option B (simpler, still effective).** Load the entire VAE for encode, unload, reload for decode.

## Implementation Plan

### Step 1: Create `GPUModuleManager` utility

New file: `cube_part/utils/gpu_manager.py`

```python
class GPUModuleManager:
    """Lazily loads a torch.nn.Module to GPU and can move it back to CPU."""

    def __init__(self, module_fn, device):
        """module_fn: callable() -> nn.Module (called once on first .to_gpu())"""
        self._module_fn = module_fn
        self._module = None
        self._device = device
        self._on_gpu = False

    @property
    def module(self):
        if self._module is None:
            self._module = self._module_fn()
        return self._module

    def to_gpu(self):
        if not self._on_gpu:
            self.module.to(self._device)
            self._on_gpu = True

    def to_cpu(self):
        if self._on_gpu:
            self._module.cpu()
            torch.cuda.empty_cache()
            self._on_gpu = False

    def __getattr__(self, name):
        return getattr(self.module, name)
```

### Step 2: Modify `ShapeDenoiserSystem.configure()` to NOT `.to(device)` immediately

In `cube_part/systems/shape_denoiser.py`:
- Remove the `.to(device)` from `ShapeDenoiserPipeline.__init__()` (line 72)
- Instead, build each sub-component on CPU, then wrap them in `GPUModuleManager`

### Step 3: Modify `ShapeDenoiserPipeline` to use phased load/unload

In `cube_part/pipelines/shape_denoiser.py`:

- Store managers for `shape_model`, `base_model`, `diffusion_model` instead of direct references
- In `encode_shape()`: `shape_model_mgr.to_gpu()` before call, `.to_cpu()` after
- In `input_to_part_shape()`:
  - Phase 1: `base_model_mgr.to_gpu()` -> encode text -> `.to_cpu()`
  - Phase 2: `diffusion_model_mgr.to_gpu()` -> diffusion loop -> `.to_cpu()`
  - Phase 3: `shape_model_mgr.to_gpu()` -> decode + extract geometry -> `.to_cpu()`

### Step 4: Modify `ShapeDenoiserSystem` to work without `.to(device)` at init

The current code does `ShapeDenoiserSystem(config.system).eval().to(device)` which moves everything at once.
Change to:
- Build system on CPU (`cpu()` or leave default)
- Wrap sub-modules after configure

### Step 5: Wire into `gradio_demo.py`

The `_build_runner()` closure (line 344-421) currently calls pipeline methods directly.
With the phased approach, the pipeline's own methods handle the load/unload internally,
so **gradio_demo.py needs minimal changes** — mainly:
- The pipeline constructor needs to accept a flag like `low_vram=True`
- Or we create a new `LowVRAMPartShapeDenoiserPipeline` subclass

### Step 6: Handle edge cases

- **torch.compile**: The VAE encoder and decoder use `@torch.compile`. Compiled kernels may need to be re-cached after moving between devices. May need to disable torch.compile when using CPU<->GPU swapping, or accept slower first inference.
- **state_dict loading**: The checkpoint is loaded in `configure()` via `self.load_state_dict()`. This can remain on CPU — safetensors loads can map to CPU.
- **Text encoder Qwen3-VL**: The processor downloads the model from HuggingFace. We should preload to CPU at startup so the user doesn't wait for the download on first request.
- **Buffers**: Some models have registered buffers (e.g., `grid_query`). Moving to CPU and back should handle these transparently since they're part of the module.

## Files to Change

| File | Change |
|------|--------|
| `cube_part/utils/gpu_manager.py` | **New** — `GPUModuleManager` class |
| `cube_part/pipelines/shape_denoiser.py` | Add low_vram mode with phased load/unload in pipeline methods |
| `cube_part/systems/shape_denoiser.py` | Allow building on CPU, expose sub-modules for wrapping |
| `examples/gradio_demo.py` | Pass `--low-vram` flag or enable by default for gradio |

## Testing / Validation

1. Launch gradio demo with `--low-vram` flag
2. Monitor VRAM with `nvidia-smi` during each phase:
   - Idle (after startup): should be near 0 (only CPU RAM used for models)
   - During encode: peak should be ~2-3 GB
   - During text encode: peak should be ~8 GB
   - During diffusion loop: peak should be ~8-10 GB (model + activation tensors)
   - During decode: peak should be ~3-5 GB
3. Verify output quality matches the non-low-vram version (same seed, same params)

## Risks & Tradeoffs

1. **Slower per-request**: Each inference now pays the cost of moving models between CPU<->GPU (PCIe bandwidth). For a 3090 (PCIe 3.0 x16, ~12 GB/s), moving 8 GB takes ~0.7 seconds. With 4 load/unload cycles, that's ~2-3 seconds overhead per request.
2. **torch.compile invalidation**: Compiled models may lose their compiled kernels when moved between devices. May need to disable compile or accept recompilation cost.
3. **Shared CPU RAM**: With 3 models on CPU, total CPU RAM usage will be ~16-20 GB. Ensure the machine has enough system RAM.
4. **Concurrent requests**: Gradio queues requests. If two requests overlap, the phased loading could conflict. The queue should be set to process one at a time (default `concurrency_limit=1`).

## Open Questions

- Should `--low-vram` be the default for the gradio demo, or opt-in?
  → Suggest making it the default since the user's stated goal is to run on their machine.
  → Can add `--full-vram` flag to keep models resident for users with more GPU memory.
