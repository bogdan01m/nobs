"""Full results section orchestration."""

from pathlib import Path

from plots.plot_efficiency import plot_efficiency_comparison
from plots.plot_embeddings_metrics import plot_embeddings_performance
from plots.plot_llm_metrics import plot_llm_performance
from plots.plot_vlm_metrics import plot_vlm_performance

from ..loaders import load_results
from ..tables import (
    EmbeddingsTableGenerator,
    InferenceTableGenerator,
    PowerMetricsTableGenerator,
    SummaryTableGenerator,
)


class ResultsSectionGenerator:
    """Orchestrate generation of complete results section for README."""

    def __init__(
        self, results_dir: Path = Path("results"), plot_base_path: str = "results"
    ):
        """Initialize with results directory.

        Args:
            results_dir: Directory containing benchmark result JSON files
            plot_base_path: Base path for plot references in markdown (default: "results")
        """
        self.results_dir = results_dir
        self.results = load_results(results_dir)
        self.plot_base_path = plot_base_path

    def _plot_path(self, filename: str) -> str:
        """Generate plot path with proper base path handling.

        Args:
            filename: Plot filename (e.g., "embeddings_performance.png")

        Returns:
            Full path to plot file
        """
        if self.plot_base_path:
            return f"{self.plot_base_path}/plots/{filename}"
        return f"plots/{filename}"

    def generate(self, include_plots: bool = True, summary_only: bool = False) -> str:
        """Generate complete results section.

        Args:
            include_plots: Whether to include plots/visualizations
            summary_only: If True, only generate summary table (for README)

        Returns:
            Complete markdown section with tables and optionally plots
        """
        if not self.results:
            return self._empty_results_message()

        if summary_only:
            # For README: only header + summary table
            sections = [
                self._generate_header(),
                self._generate_summary(include_efficiency_plots=False),
            ]
        else:
            # For docs/results.md: everything with plots
            sections = [
                self._generate_header(),
                self._generate_summary(include_efficiency_plots=include_plots),
                self._generate_power_metrics(),
                self._generate_embeddings(include_plots=include_plots),
                self._generate_llms(include_plots=include_plots),
                self._generate_vlms(include_plots=include_plots),
                self._generate_footer(),
            ]

        return "\n".join(filter(None, sections))

    def _empty_results_message(self) -> str:
        """Generate message when no results are available.

        Returns:
            Markdown message string
        """
        return "## Benchmark Results\n\n_No results available yet. Run benchmarks with `uv run python main.py`_\n"

    def _generate_header(self) -> str:
        """Generate header with timestamp.

        Returns:
            Header section string
        """
        latest = max(self.results, key=lambda x: x["timestamp"])
        timestamp = latest["timestamp"].split("T")[0]

        return "## Benchmark Results\n\n" f"> **Last Updated**: {timestamp}\n"

    def _generate_summary(self, include_efficiency_plots: bool = True) -> str:
        """Generate summary ranking table.

        Args:
            include_efficiency_plots: Whether to include efficiency plots after summary (deprecated, now ignored)

        Returns:
            Summary table markdown
        """
        summary = SummaryTableGenerator(self.results).generate()
        return summary

    def _generate_power_metrics(self) -> str:
        """Generate power metrics table if available.

        Returns:
            Power metrics section or empty string
        """
        power_section = PowerMetricsTableGenerator(self.results).generate()
        if power_section:
            return "\n" + power_section
        return ""

    def _generate_embeddings(self, include_plots: bool = True) -> str:
        """Generate embeddings section with table and optionally plots.

        Args:
            include_plots: Whether to include performance plots

        Returns:
            Embeddings section markdown or empty string
        """
        table = EmbeddingsTableGenerator(self.results).generate()
        if not table:
            return ""

        sections = ["\n### Embeddings\n", table]

        # Add embeddings performance plot if requested
        if include_plots:
            has_embeddings = any(
                any(t["task"] == "embeddings" for t in r["tasks"]) for r in self.results
            )

            if has_embeddings:
                try:
                    plot_embeddings_performance(self.results_dir)
                    sections.append(
                        f"![Embeddings Performance Profile]({self._plot_path('embeddings_performance.png')})\n"
                    )
                    sections.append(
                        "*Throughput comparison for different embedding models across hardware. "
                        "Higher values indicate better performance.*\n"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate embeddings plot: {e}")

                # Add embeddings efficiency plot
                try:
                    plot_efficiency_comparison(self.results_dir)
                    sections.append(
                        f"\n![Embeddings Efficiency]({self._plot_path('efficiency_embeddings.png')})\n"
                    )
                    sections.append(
                        "*Embeddings efficiency (RPS/W) across devices. "
                        "Higher values indicate better performance per watt.*\n"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate embeddings efficiency plot: {e}")

        return "\n".join(sections)

    def _generate_llms(self, include_plots: bool = True) -> str:
        """Generate LLMs section with table and optionally plots.

        Args:
            include_plots: Whether to include performance plots

        Returns:
            LLMs section markdown or empty string
        """
        table = InferenceTableGenerator(self.results, "llms").generate()
        if not table:
            return ""

        sections = ["\n### LLMs\n", table]

        # Add performance plots if requested
        if include_plots:
            has_llms = any(
                any(t["task"] == "llms" for t in r["tasks"]) for r in self.results
            )

            if has_llms:
                # Add performance plots
                try:
                    plot_llm_performance(self.results_dir)
                    sections.append(
                        f"![LLM E2E Latency Performance]({self._plot_path('llm_latency.png')})\n"
                    )
                    sections.append(
                        "*End-to-End Latency P50 - Lower is better. "
                        "Measures full request-to-response time.*\n\n"
                    )
                    sections.append(
                        f"![LLM Throughput Performance]({self._plot_path('llm_tps.png')})\n"
                    )
                    sections.append(
                        "*Token Generation per second (TPS) - Higher is better. "
                        "Measures token generation speed.*\n"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate LLM plots: {e}")

                # Add LLM efficiency plot
                try:
                    plot_efficiency_comparison(self.results_dir)
                    sections.append(
                        f"\n![LLM Efficiency]({self._plot_path('efficiency_llm.png')})\n"
                    )
                    sections.append(
                        "*LLM inference efficiency (TPS/W) by backend. "
                        "Higher values indicate better performance per watt.*\n"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate LLM efficiency plot: {e}")

        return "\n".join(sections)

    def _generate_vlms(self, include_plots: bool = True) -> str:
        """Generate VLMs section with table and optionally plots.

        Args:
            include_plots: Whether to include performance plots

        Returns:
            VLMs section markdown or empty string
        """
        table = InferenceTableGenerator(self.results, "vlms").generate()
        if not table:
            return ""

        sections = ["\n### VLMs\n", table]

        # Add performance plots if requested
        if include_plots:
            has_vlms = any(
                any(t["task"] == "vlms" for t in r["tasks"]) for r in self.results
            )

            if has_vlms:
                # Add performance plots
                try:
                    plot_vlm_performance(self.results_dir)
                    sections.append(
                        f"![VLM E2E Latency Performance]({self._plot_path('vlm_latency.png')})\n"
                    )
                    sections.append(
                        "*End-to-End Latency P50 - Lower is better. "
                        "Measures full request-to-response time.*\n\n"
                    )
                    sections.append(
                        f"![VLM Throughput Performance]({self._plot_path('vlm_tps.png')})\n"
                    )
                    sections.append(
                        "*Token Generation per second (TPS) - Higher is better. "
                        "Measures token generation speed.*\n"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate VLM plots: {e}")

                # Add VLM efficiency plot
                try:
                    plot_efficiency_comparison(self.results_dir)
                    sections.append(
                        f"\n![VLM Efficiency]({self._plot_path('efficiency_vlm.png')})\n"
                    )
                    sections.append(
                        "*VLM inference efficiency (TPS/W) by backend. "
                        "Higher values indicate better performance per watt.*\n"
                    )
                except Exception as e:
                    print(f"⚠️  Failed to generate VLM efficiency plot: {e}")

        return "\n".join(sections)

    def _generate_footer(self) -> str:
        """Generate footer with notes.

        Returns:
            Footer section string
        """
        return (
            "\n---\n"
            "_All metrics are shown as mean ± standard deviation across 3 runs. "
        )
