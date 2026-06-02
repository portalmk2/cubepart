# Modifications from upstream

This document records all changes made in this fork relative to the upstream
repo [`Roblox/cube`](https://github.com/Roblox/cube).

I am only interested in the parts generation model so only `cubepart` is kept.

## Directory restructuring

The original repo contained two separate codebases: `cube3d/` (Cube 3D v0.5
text-to-3D generation) and `cubepart/` (CubePart multi-part decomposition).
This fork strips `cube3d/` and its associated resources, promotes `cubepart/`
to the repository root, and drops files that were only relevant to the
monolithic `cube` project:

- `cubepart/*` → repo root (files moved up one level)
- Removed `cube3d/` (Cube 3D v0.5 code)
- Removed `cubepart/` (now empty — contents promoted to root)
- Removed `resources/` (Cube 3D teasers and demo GIFs)
- Removed `SECURITY.md`, `setup.py`

## New features

### Low-VRAM mode (`--low-vram`)

An optional inference mode that stages model loading/unloading on-demand to
reduce peak GPU memory usage. When enabled:

- Models are kept on CPU by default
- Each sub-module (shape VAE, denoiser) is moved to GPU only during its
  computation phase
- GPU memory is freed (`torch.cuda.empty_cache()`) immediately after each
  stage completes

This is controlled by the `low_vram` parameter on
`PartShapeDenoiserPipeline` and exposed via the `--low-vram` CLI / Gradio
flag.

**Files:**
- [`cube_part/utils/gpu_manager.py`](cube_part/utils/gpu_manager.py) — new;
  `on_device()` context manager for phased GPU loading
- `cube_part/pipelines/shape_denoiser.py` — `low_vram` plumbing in
  pipeline constructor and `__call__`
- `cube_part/pipelines/base.py` — `low_vram` plumbing in base pipeline
- `cube_part/systems/shape_denoiser.py` — `on_device` usage around VAE
  encode/decode and denoiser forward pass

### Fixes

#### Evaluator cache device mismatch

The evaluator cache (`_last_eval`) retained GPU tensors across inference
calls. When the same pipeline instance was reused for a second input, the
cached evaluator state lived on a different device than the freshly loaded
model, causing a cross-device error. Fixed by moving the evaluator state to
CPU after each use instead of keeping it on GPU.

#### VRAM leak in evaluator

The evaluator's intermediate tensors (not just `_last_eval`) stayed on GPU
between calls. Fixed by ensuring all evaluator-internal GPU tensors are
explicitly cleaned up after each inference pass.

## Configuration & packaging

- `pyproject.toml`: renamed package from `cube` to `cube_part`, bumped
  minimum Python from 3.7 to 3.10, added MIT license and `readme`
- `.gitignore`: replaced with Visual-Studio-style gitignore
- `.gitattributes`: replaced with Roblox LFS rules
- `CODEOWNERS`: added team reference to `@Roblox/aicube`

## Upstream tag

Last synced from upstream: [`v0.1`](https://github.com/Roblox/cube/tree/v0.1)
