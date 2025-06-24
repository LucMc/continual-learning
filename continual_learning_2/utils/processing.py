from collections.abc import Sequence
from typing import Literal

import numpy as np
from grain import python as grain


class Normalize(grain.MapTransform):
    def __init__(self, mean: float | Sequence[float], std: float | Sequence[float]):
        self.mean = mean
        self.std = std

    def map(self, element):
        mean = self.mean
        print("Normalisation: TODO - check that mean / std broadcast fine to HWC")
        breakpoint()
        if isinstance(self.mean, float):
            mean = np.broadcast_to(self.mean, element.shape)
        std = self.std
        if isinstance(self.std, float):
            std = np.broadcast_to(self.std, element.shape)
        return (element - mean) / std


class RandomCrop(grain.RandomMapTransform):
    def __init__(
        self,
        size: int,
        padding: int | Sequence[int] | None = None,
        pad_if_needed: bool = False,
        fill: int | Sequence[int] = 0,
        padding_mode: Literal["constant", "edge", "reflect", "symmetric"] = "constant",
    ):
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def random_map(self, element, rng: np.random.Generator):
        # TODO:
        return element


class RandomRotation(grain.RandomMapTransform):
    def __init__(self, degrees: Sequence[int]):
        if not degrees[1] > degrees[0]:
            raise ValueError("Degrees must specify a valid range.")
        self.degrees = degrees

    def random_map(self, element, rng: np.random.Generator):
        angle = rng.uniform(self.degrees[0], self.degrees[1]) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        h, w, _ = element.shape
        center = np.array([h // 2, w // 2])
        rotated = np.zeros_like(element)

        for i in range(h):
            for j in range(w):
                coords = np.array([i, j]) - center
                new_coords = rotation_matrix @ coords + center
                ni, nj = int(round(new_coords[0])), int(round(new_coords[1]))
                if 0 <= ni < h and 0 <= nj < w:
                    rotated[i, j] = element[ni, nj]

        return rotated


class RandomHorizontalFlip(grain.RandomMapTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def random_map(self, element, rng: np.random.Generator):  # pyright: ignore[reportIncompatibleMethodOverride]
        if rng.random() < self.p:
            return np.fliplr(element)
        return element
