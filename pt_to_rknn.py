from pathlib import Path

from ultralytics import YOLO


# ============================================================
# 配置
# ============================================================

MODELS_DIR = Path("models")

IMAGE_SIZE = 640

MODEL_NAMES = [
    "11s",
    "11m",
    "11l",
]


# ============================================================
# 依次转换
# ============================================================

for model_name in MODEL_NAMES:

    print()
    print("=" * 80)
    print(f"开始转换 YOLO{model_name} -> RKNN")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. best.pt 路径
    # --------------------------------------------------------

    pt_path = (
        MODELS_DIR /
        f"{model_name}_berry_best.pt"
    )

    if not pt_path.exists():
        print(f"[跳过] 找不到模型: {pt_path}")
        continue

    print(f"PyTorch model: {pt_path}")


    # --------------------------------------------------------
    # 2. 加载模型
    # --------------------------------------------------------

    model = YOLO(str(pt_path))


    # --------------------------------------------------------
    # 3. 导出 RKNN
    #
    # RK3588:
    # quantize=16 -> FP16
    # quantize=8  -> INT8
    #
    # 你追求准确率，所以这里使用 FP16
    # --------------------------------------------------------

    rknn_dir = model.export(
        format="rknn",

        name="rk3588",

        imgsz=IMAGE_SIZE,

        batch=1,

        quantize=16,
    )


    # --------------------------------------------------------
    # 4. 输出结果
    # --------------------------------------------------------

    rknn_dir = Path(rknn_dir)

    print(f"RKNN 目录: {rknn_dir}")

    rknn_files = list(
        rknn_dir.glob("*.rknn")
    )

    if len(rknn_files) == 0:
        print(
            f"[警告] {rknn_dir} 中没有找到 .rknn 文件"
        )
    else:
        for rknn_file in rknn_files:
            print(
                f"RKNN 文件: {rknn_file}"
            )

    print(
        f"YOLO{model_name} 转换完成"
    )


print()
print("=" * 80)
print("全部 RKNN 模型转换完成")
print("=" * 80)