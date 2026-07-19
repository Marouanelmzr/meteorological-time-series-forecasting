from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt


class BasePlots(ABC):

    def __init__(self, save_dir):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str):

        plt.tight_layout()
        plt.savefig(
            self.save_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    @abstractmethod
    def save_all(self):
        pass