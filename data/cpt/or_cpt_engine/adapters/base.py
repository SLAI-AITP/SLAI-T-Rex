from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseORGeneratorAdapter(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, seed: int, difficulty_config: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def extract_model_info(self, raw_output: Any) -> dict[str, Any]:
        raise NotImplementedError
