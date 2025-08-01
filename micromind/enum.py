from dataclasses import dataclass

@dataclass
class Stage:
    """enum to track training stage"""

    train: int = 0
    val: int = 1
    test: int = 2