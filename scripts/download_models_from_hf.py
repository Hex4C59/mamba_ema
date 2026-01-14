"""从 Hugging Face 下载语音表征 / ASR 大模型到全局目录并创建软连接。

工作流程：
1. 检查全局目录是否已有模型权重
2. 如果有，直接创建软连接到项目 pretrained_model 目录
3. 如果没有，先下载到全局目录，再创建软连接

全局目录：/mnt/shareEEx/liuyang/resources/pretrained_model
项目目录：<project_root>/pretrained_model

使用方式：
export HF_TOKEN=your_token_here
export HF_ENDPOINT=https://hf-mirror.com
uv run python scripts/download_models_from_hf.py -m wavlm-large-finetuned-iemocap --force
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_PRETRAINED = Path("/mnt/shareEEx/liuyang/resources/pretrained_model")
PROJECT_PRETRAINED = PROJECT_ROOT / "pretrained_model"
HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_TOKEN = None  # 请通过环境变量 HF_TOKEN 设置

os.environ["HF_ENDPOINT"] = HF_ENDPOINT


MODEL_CONFIGS: Dict[str, str] = {
    "wav2vec2-large-robust-12-ft-emotion-msp-dim": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "wavlm-large-msp-podcast-emotion-dim": "tiantiaf/wavlm-large-msp-podcast-emotion-dim",
    "wav2vec2-large-superb-er": "superb/wav2vec2-large-superb-er",
    "hubert-large-superb-er": "superb/hubert-large-superb-er",
    "wavlm-large-finetuned-iemocap": "Zahra99/wavlm-large-finetuned-iemocap"
}


def _get_token() -> Optional[str]:
    """获取 HuggingFace token，优先使用环境变量。"""
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or DEFAULT_HF_TOKEN
    )


def _get_cli_base() -> list[str]:
    """优先使用 hf download，若不存在则回退 huggingface-cli download。"""
    if shutil.which("hf"):
        return ["hf", "download"]
    try:
        importlib.import_module("huggingface_hub.commands.hf")
        return [sys.executable, "-m", "huggingface_hub.commands.hf", "download"]
    except ModuleNotFoundError:
        pass
    if shutil.which("huggingface-cli"):
        return ["huggingface-cli", "download"]
    raise FileNotFoundError("未找到 hf 或 huggingface-cli，请安装 huggingface_hub")


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


def _download_to_global(model_key: str, hf_id: str, force: bool = False) -> Path:
    """下载模型到全局目录并返回路径。"""
    global_dir = GLOBAL_PRETRAINED / model_key

    if global_dir.exists() and not force:
        if global_dir.is_dir() and any(global_dir.iterdir()):
            print(f"[skip] 全局目录 {global_dir} 已存在，跳过下载")
            return global_dir

    global_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] 开始下载 {model_key} ({hf_id}) 到 {global_dir}")

    cmd = _get_cli_base() + [hf_id, "--local-dir", str(global_dir)]
    if force:
        cmd.append("--force-download")

    token = _get_token()
    if token:
        cmd.extend(["--token", token])

    env = os.environ.copy()
    env["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", HF_ENDPOINT)
    if token:
        env["HF_TOKEN"] = token
    env["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"下载失败: {hf_id}, 返回码 {result.returncode}")

    print(f"[done] {model_key} 下载完成，已保存到 {global_dir}")
    return global_dir


def download_and_link(model_key: str, hf_id: str, force: bool = False) -> None:
    """下载模型到全局目录并创建软连接到项目目录。

    参数
    ----
    model_key:
        本地模型目录名（如 wavlm_large）。
    hf_id:
        Hugging Face 模型 ID（如 microsoft/wavlm-large）。
    force:
        是否强制重新下载（即使全局目录已存在）。
    """
    global_dir = GLOBAL_PRETRAINED / model_key
    project_link = PROJECT_PRETRAINED / model_key

    # 检查全局目录是否存在
    if global_dir.exists() and global_dir.is_dir() and any(global_dir.iterdir()):
        print(f"[found] 在全局目录找到模型: {global_dir}")
        if not force:
            _create_symlink(global_dir, project_link)
            return

    # 下载到全局目录
    global_dir = _download_to_global(model_key, hf_id, force=force)

    # 创建软连接
    _create_symlink(global_dir, project_link)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Hugging Face 下载模型到全局目录并创建软连接"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="自定义 Hugging Face 模型 ID",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="本地保存目录名（仅在指定 --repo-id 时使用）",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["all", *MODEL_CONFIGS.keys()],
        default="all",
        help="要下载的模型（默认: all）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使全局目录已存在）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    GLOBAL_PRETRAINED.mkdir(parents=True, exist_ok=True)
    PROJECT_PRETRAINED.mkdir(parents=True, exist_ok=True)

    if args.repo_id:
        local_name = args.name or args.repo_id.split("/")[-1]
        download_and_link(local_name, args.repo_id, force=bool(args.force))
        return

    if args.model == "all":
        for key, hf_id in MODEL_CONFIGS.items():
            download_and_link(key, hf_id, force=bool(args.force))
    else:
        hf_id = MODEL_CONFIGS[args.model]
        download_and_link(args.model, hf_id, force=bool(args.force))


if __name__ == "__main__":
    main()
