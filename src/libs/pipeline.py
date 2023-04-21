"""Pipeline class"""

from abc import ABC, abstractmethod


class MLPipeline(ABC):
    def train(self) -> None:
        pass
