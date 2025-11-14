"""Embeddings performance table generator."""

from typing import Any

from ..extractors import ModelMetricsExtractor, SummaryMetricsExtractor
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

        lines = [
            "#### Text Embeddings (3000 IMDB samples)\n",
            "_RPS = Rows Per Second — number of text samples encoded per second._\n",
        ]
        lines.extend(
            self._build_header(
                [
                    "Device",
                    "Model",
                    "RPS (mean ± std)",
                    "Time (s) (mean ± std)",
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

                # Helper function to format value with std
                def fmt_with_std(val, std_val):
                    if isinstance(val, (int, float)) and isinstance(
                        std_val, (int, float)
                    ):
                        return f"{val:.2f} ± {std_val:.2f}"
                    elif isinstance(val, (int, float)):
                        return f"{val:.2f}"
                    return "-"

                # Get mean ± std for RPS and Time
                # Simple approach: mean(run1, run2, run3) ± std(run1, run2, run3)
                rps_mean = model_data.get("final_mean_rps")
                rps_std = model_data.get("final_std_rps")
                time_mean = model_data.get("final_mean_e2e_latency_s")
                time_std = model_data.get("final_std_e2e_latency_s")

                rows.append(
                    f"| {device} | {model_name} | "
                    f"{fmt_with_std(rps_mean, rps_std)} | "
                    f"{fmt_with_std(time_mean, time_std)} | "
                    f"{model_data['embedding_dimension']} | "
                    f"{model_data['batch_size']} |"
                )

        return rows


class EmbeddingsEfficiencyTableGenerator(BaseTableGenerator):
    """Generate embeddings performance per watt table."""

    def generate(self) -> str:
        """Generate embeddings efficiency table.

        Returns:
            Markdown table string or empty string if no data
        """
        if not self._has_data():
            return ""

        metrics_results = self._extract_metrics()
        if not metrics_results:
            return ""

        lines = [
            "\n#### Embeddings Performance per Watt (GPU)\n",
            "_Higher values indicate better performance per watt._\n",
        ]
        lines.extend(
            self._build_header(
                [
                    "Device",
                    "RPS P50",
                    "GPU Power P50 (W)",
                    "Efficiency (RPS/W)",
                ]
            )
        )

        for item in metrics_results:
            lines.append(self._format_row(item))

        return "\n".join(lines) + "\n"

    def _extract_metrics(self) -> list[dict[str, Any]]:
        """Extract efficiency metrics for each result.

        Returns:
            List of dicts with result and efficiency metrics
        """
        metrics_results = []
        extractor = SummaryMetricsExtractor()

        for result in self.results:
            embeddings_rps = extractor.extract_embeddings_rps_p50(result)
            gpu_watts = extractor.extract_gpu_watts_p50(result)

            # Only include results with both metrics
            if embeddings_rps and gpu_watts:
                efficiency = embeddings_rps / gpu_watts
                metrics_results.append(
                    {
                        "device": result["device_info"]["host"],
                        "rps": embeddings_rps,
                        "gpu_watts": gpu_watts,
                        "efficiency": efficiency,
                    }
                )

        return metrics_results

    def _format_row(self, item: dict[str, Any]) -> str:
        """Format a single efficiency table row.

        Args:
            item: Dict with device and efficiency metrics

        Returns:
            Markdown table row string
        """
        return (
            f"| {item['device']} | "
            f"{item['rps']:.1f} | "
            f"{item['gpu_watts']:.1f} | "
            f"{item['efficiency']:.2f} |"
        )
