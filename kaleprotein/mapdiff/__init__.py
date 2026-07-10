"""MapDiff, exposed through the same auto API."""

from kaleprotein.mapdiff.data import MapDiffDataModule
from kaleprotein.mapdiff.modeling import MapDiffForInverseFolding

__all__ = ["MapDiffDataModule", "MapDiffForInverseFolding"]
