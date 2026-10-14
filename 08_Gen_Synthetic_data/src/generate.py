"""Stable Diffusion 1.5 텍스트→이미지 실험 (A1 guidance, A2 fp32/fp16, A3 steps 비교).

노트북 stable_diffusion_inpainting_pcb.ipynb의 실험 A1, A2, A3 셀을 함수로 옮긴 것. GPU 필요.
"""
import gc
import time

import torch
from diffusers import StableDiffusionPipeline

MODEL_IDS = ["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5"]
PROMPT = "A photo of man riding a horse in the moon wearing astronaut outfits"


def load_pipe(dtype=torch.float16):
    # runwayml 저장소가 내려간 경우를 대비해 미러를 차례로 시도
    for mid in MODEL_IDS:
        try:
            return StableDiffusionPipeline.from_pretrained(mid, torch_dtype=dtype).to("cuda")
        except Exception as e:
            print("failed:", mid, type(e).__name__)
    raise RuntimeError("Stable Diffusion 1.5를 불러오지 못함")


def timed_generate(pipe, prompt, guidance, seed=42, steps=30):
    """시드를 고정해 한 장 생성하고 걸린 시간(초)을 함께 돌려준다."""
    gen = torch.Generator("cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t = time.time()
    img = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]
    torch.cuda.synchronize()
    return img, time.time() - t


def compare_guidance(prompt=PROMPT, scales=(1, 7.5, 20), seed=42, steps=30):
    """A1: guidance만 바꿔 생성. 계산량은 같으므로 시간도 같아야 한다."""
    pipe = load_pipe(torch.float16)
    pipe(prompt, num_inference_steps=2)  # 첫 호출 준비 시간 제외
    results = {}
    for g in scales:
        img, sec = timed_generate(pipe, prompt, g, seed, steps)
        print(f"guidance={g:>4}  time={sec:.2f}s")
        results[g] = img
    return results


def compare_steps(prompt=PROMPT, steps_list=(10, 20, 30, 50), guidance=7.5, seed=42):
    """A3: 노이즈 제거 단계 수만 바꿔 생성. 시간은 단계 수에 비례해야 한다."""
    pipe = load_pipe(torch.float16)
    pipe(prompt, num_inference_steps=2)
    results = {}
    for n in steps_list:
        img, sec = timed_generate(pipe, prompt, guidance, seed, n)
        print(f"steps={n:2d}  time={sec:.2f}s")
        results[n] = img
    return results


def compare_dtype(prompt=PROMPT, guidance=7.5, seed=42, steps=30):
    """A2: fp32와 fp16의 생성 시간과 최대 GPU 메모리(GB)를 잰다."""
    results = {}
    for name, dt in [("fp32", torch.float32), ("fp16", torch.float16)]:
        pipe = load_pipe(dt)
        pipe(prompt, num_inference_steps=2)
        torch.cuda.reset_peak_memory_stats()
        img, sec = timed_generate(pipe, prompt, guidance, seed, steps)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{name}: time={sec:.2f}s  peak_GPU_memory={mem:.2f}GB")
        results[name] = (img, sec, mem)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "guidance"
    out = {"guidance": compare_guidance, "dtype": compare_dtype, "steps": compare_steps}[which]()
    for k, v in out.items():
        (v[0] if isinstance(v, tuple) else v).save(f"{which}_{k}.png")
