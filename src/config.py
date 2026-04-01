from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    """加载 config.yaml 并解析路径"""
    config_path = PROJECT_ROOT / 'config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # 解析相对路径为绝对路径
    cfg['tts']['model_dir'] = str(PROJECT_ROOT / cfg['tts']['model_dir'])
    cfg['tts']['speaker_wav'] = str(PROJECT_ROOT / cfg['tts']['speaker_wav'])
    cfg['paths']['resources_dir'] = PROJECT_ROOT / cfg['paths']['resources_dir']
    cfg['paths']['output_dir'] = PROJECT_ROOT / cfg['paths']['output_dir']

    return cfg


# 全局配置对象
config = load_config()
