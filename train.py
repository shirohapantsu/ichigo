import gc
import shutil
from pathlib import Path
import albumentations as A
import cv2 as cv
import torch
from ultralytics import YOLO


# ============================================================
# 基础配置
# ============================================================

DATA_PATH = "Strawberry-Pose-Detection.v2i.yolov8/data.yaml"

MODELS_DIR = Path("models")

TEST_IMAGE = "实机.png"

IMAGE_SIZE = 640

WORKERS = 4

EPOCHS = 200

PATIENCE = 75

# ============================================================
# 三个模型分别设置 batch
# ============================================================

MODEL_CONFIGS = {
    "11s": {
        "path": MODELS_DIR / "yolo11s-pose.pt",
        "batch": 8,
    },

    "11m": {
        "path": MODELS_DIR / "yolo11m-pose.pt",
        "batch": 4,
    },

    "11l": {
        "path": MODELS_DIR / "yolo11l-pose.pt",
        "batch": 4,
    },
}


custom_augmentations = [

    # ----------------------------------------
    # 模糊
    # 总概率约 30%
    # ----------------------------------------
    A.OneOf(
        [
            A.Blur(
                blur_limit=(3, 5),
                p=1.0,
            ),

            A.GaussianBlur(
                blur_limit=(3, 7),
                p=1.0,
            ),

            A.MotionBlur(
                blur_limit=(3, 7),
                p=1.0,
            ),
        ],
        p=0.30,
    ),

    # ----------------------------------------
    # 传感器/低光环境噪声
    # ----------------------------------------
    A.GaussNoise(
        std_range=(0.01, 0.035),
        p=0.15,
    ),

    # ----------------------------------------
    # 光照和对比度变化
    # ----------------------------------------
    A.RandomBrightnessContrast(
        brightness_limit=0.20,
        contrast_limit=0.20,
        p=0.30,
    ),

    # ----------------------------------------
    # 局部对比度增强
    # 模拟不同摄像头 ISP
    # ----------------------------------------
    A.CLAHE(
        clip_limit=(1.0, 3.0),
        p=0.10,
    ),
]

# ============================================================
# 依次训练
# ============================================================

for model_name, config in MODEL_CONFIGS.items():

    print("=" * 80)
    print(f"开始训练 YOLO{model_name}")
    print(f"模型路径: {config['path']}")
    print(f"Batch Size: {config['batch']}")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. 加载预训练模型
    # --------------------------------------------------------

    model = YOLO(str(config["path"]))

    model.info()


    # --------------------------------------------------------
    # 2. 训练
    # --------------------------------------------------------

    model.train(
        data=DATA_PATH,

        epochs=EPOCHS,
        patience=PATIENCE,

        imgsz=IMAGE_SIZE,

        rect=False,
        multi_scale=0.1,

        batch=config["batch"],
        workers=WORKERS,

        mosaic=1.0,
        mixup=0.10,
        close_mosaic=25,

        translate=0.12,
        scale=0.55,
        degrees=8.0,
        shear=2.0,
        perspective=0.0003,

        hsv_h=0.02,
        hsv_s=0.60,
        hsv_v=0.45,

        device=0,
        amp=True,

        # 自定义增强
        augmentations=custom_augmentations,

        # 分别保存，避免三个模型混到一起
        project="runs/berry",
        name=model_name,

        save = True
    )


    # --------------------------------------------------------
    # 3. 获取当前模型的 best.pt
    # --------------------------------------------------------

    best_path = Path(model.trainer.best)

    print(f"\nYOLO{model_name} best.pt:")
    print(best_path)


    # --------------------------------------------------------
    # 4. 重新加载 best.pt
    # --------------------------------------------------------

    best_model = YOLO(str(best_path))


    # --------------------------------------------------------
    # 5. 用 test.jpg 测试
    # --------------------------------------------------------

    results = best_model.predict(
        source=TEST_IMAGE,
        imgsz=IMAGE_SIZE,
        conf=0.25,
        verbose=False,
    )

    for result in results:

        detected_image = result.plot()

        output_image = f"test_{model_name}.jpg"

        cv.imwrite(
            output_image,
            detected_image
        )

        print(f"测试图片已保存: {output_image}")


    # --------------------------------------------------------
    # 6. 保存一份 best.pt 到 models/
    # --------------------------------------------------------

    best_destination = (
        MODELS_DIR /
        f"{model_name}_berry_best.pt"
    )

    shutil.copy2(
        best_path,
        best_destination
    )

    print(
        f"best.pt 已复制到: "
        f"{best_destination}"
    )


    # --------------------------------------------------------
    # 7. 清理显存和内存
    # --------------------------------------------------------

    del best_model
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"YOLO{model_name} 训练完成\n")


print("=" * 80)
print("YOLO11s / YOLO11m / YOLO11l 全部训练完成")
print("=" * 80)