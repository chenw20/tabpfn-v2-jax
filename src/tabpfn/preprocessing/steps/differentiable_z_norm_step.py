"""Differentiable Z-Norm Step."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingStep,
)

if TYPE_CHECKING:
    from tabpfn.preprocessing.datamodel import FeatureSchema


class DifferentiableZNormStep(PreprocessingStep):
    """Differentiable Z-Norm Step."""

    def __init__(self):
        super().__init__()

        self.means = np.array([])
        self.stds = np.array([])

    @override
    def _fit(  # type: ignore
        self,
        X: np.ndarray,
        feature_schema: FeatureSchema,
    ) -> FeatureSchema:
        self.means = X.mean(axis=0, keepdims=True)
        self.stds = X.std(axis=0, keepdims=True)
        return feature_schema

    def _transform(
        self, X: np.ndarray, *, is_test: bool = False
    ) -> tuple[np.ndarray, None, None]:
        assert X.shape[1] == self.means.shape[1]
        assert X.shape[1] == self.stds.shape[1]
        return (X - self.means) / self.stds, None, None


__all__ = [
    "DifferentiableZNormStep",
]
