"""Table generation modules."""

from .base import BaseTableGenerator
from .embeddings import EmbeddingsEfficiencyTableGenerator, EmbeddingsTableGenerator
from .gpu_groups import GPUGroupedTableGenerator
from .llm_vlm import InferenceEfficiencyTableGenerator, InferenceTableGenerator
from .power_metrics import PowerMetricsTableGenerator
from .summary import SummaryTableGenerator

__all__ = [
    "BaseTableGenerator",
    "SummaryTableGenerator",
    "PowerMetricsTableGenerator",
    "EmbeddingsTableGenerator",
    "EmbeddingsEfficiencyTableGenerator",
    "InferenceTableGenerator",
    "InferenceEfficiencyTableGenerator",
    "GPUGroupedTableGenerator",
]
