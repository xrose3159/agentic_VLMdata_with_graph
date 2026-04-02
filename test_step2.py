"""
测试修改后的 step2 VLM + SAM3 流程。

用法（需要 GPU）：
  srun -p belt_road --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    /mnt/petrelfs/shangxiaoran/anaconda3/envs/math/bin/python test_step2.py

或指定某张图：
  srun -p belt_road --gres=gpu:1 --cpus-per-task=4 --mem=32G \
    /mnt/petrelfs/shangxiaoran/anaconda3/envs/math/bin/python test_step2.py --image output/images/img_0010.jpg
"""

import argparse
import glob
import json
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from core.logging_setup import get_logger

logger = get_logger("test_step2", log_file="test_step2.log")


def main():
    parser = argparse.ArgumentParser(description="测试 step2 VLM + SAM3 流程")
    parser.add_argument("--image", type=str, default=None, help="指定测试图片路径")
    args = parser.parse_args()

    if args.image:
        img_path = args.image
    else:
        from core.config import FILTERED_IMAGE_DIR
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        img_paths = []
        for p in patterns:
            img_paths.extend(glob.glob(os.path.join(FILTERED_IMAGE_DIR, p)))
        img_paths.sort()
        if not img_paths:
            print("没有找到筛选后的图片，请先运行 step1")
            sys.exit(1)
        img_path = img_paths[0]

    img_id = os.path.splitext(os.path.basename(img_path))[0]
    print(f"\n{'='*60}")
    print(f"测试图片: {img_path}  (id={img_id})")
    print(f"{'='*60}\n")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    from step2_enrich import SAM3_MODEL_PATH, SAM3_VOCAB_PATH, SAM3_CONF_THRESHOLD

    print(f"SAM3 模型: {SAM3_MODEL_PATH}")
    print(f"  文件存在: {os.path.isfile(SAM3_MODEL_PATH)}")
    if os.path.isfile(SAM3_MODEL_PATH):
        print(f"  文件大小: {os.path.getsize(SAM3_MODEL_PATH) / 1024**3:.2f} GB")
    print(f"BPE 词汇表: {SAM3_VOCAB_PATH}")
    print(f"  文件存在: {os.path.isfile(SAM3_VOCAB_PATH)}")
    print(f"置信度阈值: {SAM3_CONF_THRESHOLD}")
    print()

    # ---------- 测试 SAM3 模型加载 ----------
    print("--- 测试 SAM3 模型加载 ---")
    from step2_enrich import _load_sam3
    t0 = time.time()
    processor = _load_sam3()
    print(f"  加载耗时: {time.time() - t0:.2f}s")
    print()

    # ---------- 测试 SAM3 Point Prompt 定位 ----------
    print("--- 测试 SAM3 Point Prompt 定位 ---")
    from PIL import Image
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    state = processor.set_image(image)
    print(f"  图片编码完成: {w}x{h}")

    test_points = [
        ("图片中心", [0.5, 0.5]),
        ("左上角", [0.15, 0.15]),
        ("右下角", [0.85, 0.85]),
    ]
    for label, (cx, cy) in test_points:
        processor.reset_all_prompts(state)
        state = processor.add_geometric_prompt(
            box=[cx, cy, 0.001, 0.001], label=True, state=state
        )
        boxes = state.get("boxes")
        scores = state.get("scores")
        n = len(boxes) if boxes is not None else 0
        print(f"  点'{label}'({cx:.2f},{cy:.2f}): {n} 个分割结果", end="")
        if n > 0:
            best = scores.argmax().item()
            bbox = boxes[best].cpu().tolist()
            conf = float(scores[best].cpu())
            area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (w * h)
            print(f", bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}], "
                  f"conf={conf:.3f}, 面积占比={area:.1%}")
        else:
            print()
    print()

    # ---------- 测试完整 VLM + SAM3 流程 ----------
    print("--- 测试完整 VLM + SAM3 流程 ---")
    from step2_enrich import extract_entities_vlm_sam
    t0 = time.time()
    result = extract_entities_vlm_sam(img_path, img_id=img_id)
    total_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"结果摘要（耗时 {total_time:.1f}s）")
    print(f"{'='*60}")
    print(f"图片描述: {result.get('image_description', '')[:200]}...")
    print(f"领域: {result.get('domain', '?')}")
    print(f"实体数量: {len(result.get('entities', []))}")
    print()

    for e in result.get("entities", []):
        bbox_str = (
            f"[{e['bbox'][0]:.0f},{e['bbox'][1]:.0f},{e['bbox'][2]:.0f},{e['bbox'][3]:.0f}]"
            if e.get("bbox") else "无"
        )
        print(
            f"  {e['id']:4s} | {e['name'][:30]:30s} | "
            f"type={e.get('type','?'):10s} | "
            f"bbox={bbox_str:24s} | "
            f"conf={e.get('confidence',0):.4f} | "
            f"source={e.get('source','?')}"
        )

    out_file = f"test_step2_result_{img_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {out_file}")


if __name__ == "__main__":
    main()
