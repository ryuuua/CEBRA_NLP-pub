from hydra import compose, initialize
from omegaconf import OmegaConf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config_schema import AppConfig, DatasetConfig


def test_dataset_source_default():
    with initialize(version_base='1.2', config_path='../conf'):
        cfg = compose(config_name='config', overrides=['dataset=imdb'])
    dataset_cfg = OmegaConf.merge(OmegaConf.structured(DatasetConfig), cfg.dataset)
    assert dataset_cfg.source == 'hf'
