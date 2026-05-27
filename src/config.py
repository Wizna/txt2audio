from pathlib import Path
import yaml

from pathing import resolve_runtime_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_config_paths(cfg):
    """将配置中的路径项规范化为绝对路径。"""
    cfg['tts']['model_dir'] = str(resolve_runtime_path(cfg['tts']['model_dir'], PROJECT_ROOT))
    cfg['tts']['speaker_wav'] = str(resolve_runtime_path(cfg['tts']['speaker_wav'], PROJECT_ROOT))
    cfg['paths']['resources_dir'] = resolve_runtime_path(cfg['paths']['resources_dir'], PROJECT_ROOT)
    cfg['paths']['output_dir'] = resolve_runtime_path(cfg['paths']['output_dir'], PROJECT_ROOT)
    return cfg


def load_config():
    """加载 config.yaml 并解析路径"""
    config_path = PROJECT_ROOT / 'config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    return normalize_config_paths(cfg)


# 全局配置对象
config = load_config()
