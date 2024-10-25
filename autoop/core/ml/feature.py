from typing import Literal
import numpy as np
from autoop.core.ml.dataset import Dataset

# Implement getter setter??
class Feature():
    def __init__(self, name: str, type: Literal["categorical", "numerical"]) -> None:
        self.name = name
        self.type = type

    def __str__(self):
        return f"Name: {self.name} | Type: {self.type}"
