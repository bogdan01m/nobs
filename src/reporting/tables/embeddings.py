"""Embeddings performance table generator."""

from typing import Any

from ..extractors import ModelMetricsExtractor
from ..formatters import format_time_with_std
from .base import BaseTableGenerator


class EmbeddingsTableGenerator(BaseTableGenerator):
    """Generate detailed embeddings performance table."""

    def generate(self) -> str:
        """Generate embeddings table.

        Returns:
            Markdown table string or empty string if no embeddings data
        """
        models = self._collect_all_models()
        if not models:
            return ""

        lines = ["#### Text Embeddings (100 IMDB samples)\n"]
        lines.extend(
            self._build_header(
                [
                    "Device",
                    "Model",
                    "Rows/sec",
                    "Time (s)",
                    "Embedding Dim",
                    "Batch Size",
                ]
            )
        )

        for result in self.results:
            lines.extend(self._format_device_rows(result, models))

        return "\n".join(lines) + "\n"

    def _collect_all_models(self) -> set[str]:
        """Get unique models across all results.

        Returns:
            Set of model names
        """
        all_models: set[str] = set()
        extractor = ModelMetricsExtractor()

        for result in self.results:
            models = extractor.extract_embeddings_models(result)
            all_models.update(models.keys())

        return all_models

    def _format_device_rows(
        self, result: dict[str, Any], all_models: set[str]
    ) -> list[str]:
        """Format rows for a single device.

        Args:
            result: Benchmark result dictionary
            all_models: Set of all model names to include

        Returns:
            List of markdown table row strings
        """
        rows: list[str] = []
        device = result["device_info"]["host"]
        extractor = ModelMetricsExtractor()
        embeddings_models = extractor.extract_embeddings_models(result)

        if not embeddings_models:
            return rows

        for model_name in sorted(all_models):
            if model_name in embeddings_models:
                model_data = embeddings_models[model_name]

                # Format rows/sec with std
                rps_median = model_data["median_rows_per_second"]
                rps_std = model_data.get("std_rows_per_second", 0)
                rps_str = format_time_with_std(rps_median, rps_std)

                # Format time with std
                time_median = model_data["median_encoding_time_seconds"]
                time_std = model_data.get("std_encoding_time_seconds", 0)
                time_str = format_time_with_std(time_median, time_std)

                rows.append(
                    f"| {device} | {model_name} | "
                    f"{rps_str} | "
                    f"{time_str} | "
                    f"{model_data['embedding_dimension']} | "
                    f"{model_data['batch_size']} |"
                )

        return rows
