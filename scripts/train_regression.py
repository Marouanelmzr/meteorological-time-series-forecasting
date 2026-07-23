from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.logging.wandb_logger import WandBLogger
from src.training.regression_trainer import RegressionTrainer


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="config",
)
def main(cfg: DictConfig):

    print("=" * 80)
    print("Configuration")
    print("=" * 80)

    print(OmegaConf.to_yaml(cfg))

    logger = None

    if cfg.wandb.enabled:

        logger = WandBLogger(cfg)

        logger.start()

    trainer = RegressionTrainer(
        cfg=cfg,
        logger=logger,
    )

    trainer.train()

    if logger is not None:

        logger.finish()


if __name__ == "__main__":

    main()