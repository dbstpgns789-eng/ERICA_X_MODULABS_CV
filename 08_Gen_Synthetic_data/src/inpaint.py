"""정상 PCB 사진에 결함 Inpainting (B1)과 마스크 바깥 원본 되붙이기 (B2).

노트북 stable_diffusion_inpainting_pcb.ipynb의 실험 B1, B2 셀을 함수로 옮긴 것.
inpaint()는 GPU가 필요하고, box_mask()와 composite()는 CPU에서 돌아간다.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# stabilityai/stable-diffusion-2-inpainting은 Hub에서 401이 나서 SD1.5 계열을 차례로 시도
INPAINT_IDS = [
    "runwayml/stable-diffusion-inpainting",
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "Lykon/dreamshaper-8-inpainting",
]
PROMPT = "a crack on the circuit board, broken trace"
NEGATIVE = "smooth, perfect, clean, blurry"


def box_mask(size, box):
    """흰색(255) = 다시 그릴 곳. box = (x0, y0, x1, y1)."""
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def load_inpaint_pipe():
    import torch
    from diffusers import AutoPipelineForInpainting
    for mid in INPAINT_IDS:
        try:
            pipe = AutoPipelineForInpainting.from_pretrained(mid, torch_dtype=torch.float16).to("cuda")
            print("loaded:", mid)
            return pipe
        except Exception as e:
            print("failed:", mid, type(e).__name__)
    raise RuntimeError("inpainting 모델을 불러오지 못함")


def inpaint(pipe, image, mask, prompt=PROMPT, negative=NEGATIVE, steps=50, guidance=7.5, seed=42):
    import torch
    gen = torch.Generator("cuda").manual_seed(seed)
    return pipe(prompt=prompt, negative_prompt=negative, image=image, mask_image=mask,
                num_inference_steps=steps, guidance_scale=guidance, generator=gen).images[0]


def composite(generated, original, mask, feather=0):
    """마스크 안은 생성 결과, 바깥은 원본 픽셀.

    Inpainting도 사진 전체를 latent로 압축했다 복원하므로 마스크 바깥의 작은 글자가
    뭉개진다. 바깥을 원본으로 되돌리면 글자는 살아나지만 경계에 직선 이음새가 생긴다.
    feather > 0이면 마스크 경계를 그만큼 흐리게 해서 두 사진을 부드럽게 섞는다.
    """
    size = generated.size
    m = mask.resize(size)
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    m = (np.asarray(m, dtype=np.float32) / 255.0)[..., None]
    gen = np.asarray(generated.convert("RGB"), dtype=np.float32)
    org = np.asarray(original.convert("RGB").resize(size), dtype=np.float32)
    return Image.fromarray((gen * m + org * (1 - m)).astype("uint8"))


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "normal.png"
    init = Image.open(src).convert("RGB").resize((512, 512))
    mask = box_mask(init.size, (200, 250, 300, 300))
    out = inpaint(load_inpaint_pipe(), init, mask)
    out.save("b1_inpainted.png")
    composite(out, init, mask).save("b2_composite.png")
    composite(out, init, mask, feather=8).save("b2_composite_feather8.png")
