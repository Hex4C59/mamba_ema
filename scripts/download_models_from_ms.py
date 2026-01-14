#!/usr/bin/env python3
"""从魔塔社区下载预训练模型到全局目录并创建软连接。

工作流程：
1. 检查全局目录是否已有模型权重
2. 如果有，直接创建软连接到项目 pretrained_model 目录
3. 如果没有，先下载到全局目录，再创建软连接

全局目录：/mnt/shareEEx/liuyang/resources/pretrained_model
项目目录：<project_root>/pretrained_model
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

from modelscope import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_PRETRAINED = Path("/mnt/shareEEx/liuyang/resources/pretrained_model")
PROJECT_PRETRAINED = PROJECT_ROOT / "pretrained_model"


MODEL_CONFIGS: Dict[str, str] = {
    "emotion2vec_plus_large": "iic/emotion2vec_plus_large",
}


def _create_symlink(source: Path, target: Path) -> None:
    """创建软连接，如果目标已存在则先删除。"""
    if target.is_symlink() or target.exists():
        if target.is_symlink():
            print(f"[remove] 删除已存在的软连接: {target}")
        else:
            print(f"[remove] 删除已存在的目录/文件: {target}")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    print(f"[symlink] 创建软连接: {target} -> {source}")


def _download_to_global(model_key: str, model_id: str) -> Path:
    """下载模型到全局目录并返回路径。"""
    global_dir = GLOBAL_PRETRAINED / model_key

    print(f"[download] 开始下载 {model_key} ({model_id}) 到全局目录")
    print(f"[download] 目标路径: {global_dir}")

    model_dir = snapshot_download(model_id=model_id, cache_dir=str(GLOBAL_PRETRAINED))

    print(f"[done] {model_key} 下载完成")
    print(f"[done] 模型路径: {model_dir}")

    print("\n下载的文件列表:")
    for root, _, files in os.walk(model_dir):
        level = root.replace(model_dir, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / (1024**3)
            print(f"{subindent}{file} ({file_size:.2f} GB)")

    return Path(model_dir)


def download_and_link(model_key: str, model_id: str) -> None:
    """下载模型到全局目录并创建软连接到项目目录。

    参数
    ----
    model_key:
        本地模型目录名（如 emotion2vec_plus_large）。
    model_id:
        ModelScope 模型 ID（如 iic/emotion2vec_plus_large）。
    """
    global_pattern = GLOBAL_PRETRAINED / "**" / model_key
    project_link = PROJECT_PRETRAINED / model_key

    # 在全局目录中查找已存在的模型
    import glob

    existing = list(Path(p) for p in glob.glob(str(global_pattern), recursive=True))

    if existing:
        global_dir = existing[0]
        if global_dir.is_dir() and any(global_dir.iterdir()):
            print(f"[found] 在全局目录找到模型: {global_dir}")
            _create_symlink(global_dir, project_link)
            return

    # 下载到全局目录
    global_dir = _download_to_global(model_key, model_id)

    # 创建软连接
    _create_symlink(global_dir, project_link)


def main() -> None:
    GLOBAL_PRETRAINED.mkdir(parents=True, exist_ok=True)
    PROJECT_PRETRAINED.mkdir(parents=True, exist_ok=True)

    for model_key, model_id in MODEL_CONFIGS.items():
        try:
            download_and_link(model_key, model_id)
        except Exception as exc:
            print(f"\n下载失败: {model_key} ({model_id})，原因: {exc}")
            raise


if __name__ == "__main__":
    main()
