"""Full results section orchestration."""

from pathlib import Path

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
from .visualizations import generate_token_metric_visualizations


class ResultsSectionGenerator:
    """Orchestrate generation of complete results section for README."""

    def __init__(self, results_dir: Path = Path("results")):
        """Initialize with results directory.

        Args:
            results_dir: Directory containing benchmark result JSON files
        """
        self.results_dir = results_dir
        self.results = load_results(results_dir)

    def generate(self) -> str:
        """Generate complete results section for README.

        Returns:
            Complete markdown section with all tables and plots
        """
        if not self.results:
            return self._empty_results_message()

        sections = [
            self._generate_header(),
            self._generate_summary(),
            self._generate_power_metrics(),
            self._generate_embeddings(),
            self._generate_llms(),
            self._generate_vlms(),
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

        return (
            "## Benchmark Results\n\n"
            f"> **Last Updated**: {timestamp}\n"
            "### 🏆 Overall Ranking\n"
        )

    def _generate_summary(self) -> str:
        """Generate summary ranking table.

        Returns:
            Summary table markdown
        """
        return SummaryTableGenerator(self.results).generate()

    def _generate_power_metrics(self) -> str:
        """Generate power metrics table if available.

        Returns:
            Power metrics section or empty string
        """
        power_section = PowerMetricsTableGenerator(self.results).generate()
        if power_section:
            return "\n" + power_section
        return ""

    def _generate_embeddings(self) -> str:
        """Generate embeddings section with table and plots.

        Returns:
            Embeddings section markdown or empty string
        """
        table = EmbeddingsTableGenerator(self.results).generate()
        if not table:
            return ""

        sections = ["\n### Embeddings\n", table]

        # Add embeddings performance plot
        has_embeddings = any(
            any(t["task"] == "embeddings" for t in r["tasks"]) for r in self.results
        )

        if has_embeddings:
            try:
                plot_embeddings_performance(self.results_dir)
                sections.append(
                    "![Embeddings Performance Profile](results/plots/embeddings_performance.png)\n"
                )
                sections.append(
                    "*Throughput comparison for different embedding models across hardware. "
                    "Higher values indicate better performance.*\n"
                )
            except Exception as e:
                print(f"⚠️  Failed to generate embeddings plot: {e}")

        return "\n".join(sections)

    def _generate_llms(self) -> str:
        """Generate LLMs section with table and plots.

        Returns:
            LLMs section markdown or empty string
        """
        table = InferenceTableGenerator(self.results, "llms").generate()
        if not table:
            return ""

        sections = ["\n### LLMs\n", table]

        # Add token metric plots
        has_llms = any(
            any(t["task"] == "llms" for t in r["tasks"]) for r in self.results
        )

        if has_llms:
            token_plots = generate_token_metric_visualizations(self.results_dir)
            llm_plots = token_plots.get("llms")

            if llm_plots:
                sections.append(
                    f"![LLM TTFT vs Input Tokens]({llm_plots['ttft'].as_posix()})\n"
                )
                sections.append(
                    "*Time To First Token across prompt lengths. Lower values mean faster first responses.*\n\n"
                )
                sections.append(
                    f"![LLM Generation Time vs Output Tokens]({llm_plots['tg'].as_posix()})\n"
                )
                sections.append(
                    "*Generation time growth relative to output length. Lower values reflect faster completions.*\n"
                )

            # Add performance plots
            try:
                plot_llm_performance(self.results_dir)
                sections.append("![LLM TTFT Performance](results/plots/llm_ttft.png)\n")
                sections.append(
                    "*Time To First Token (TTFT) - Lower is better. "
                    "Measures response latency.*\n\n"
                )
                sections.append(
                    "![LLM Throughput Performance](results/plots/llm_tps.png)\n"
                )
                sections.append(
                    "*Token Generation per second (TG) - Higher is better. "
                    "Measures token generation.*\n"
                )
            except Exception as e:
                print(f"⚠️  Failed to generate LLM plots: {e}")

        return "\n".join(sections)

    def _generate_vlms(self) -> str:
        """Generate VLMs section with table and plots.

        Returns:
            VLMs section markdown or empty string
        """
        table = InferenceTableGenerator(self.results, "vlms").generate()
        if not table:
            return ""

        sections = ["\n### VLMs\n", table]

        # Add token metric plots
        has_vlms = any(
            any(t["task"] == "vlms" for t in r["tasks"]) for r in self.results
        )

        if has_vlms:
            token_plots = generate_token_metric_visualizations(self.results_dir)
            vlm_plots = token_plots.get("vlms")

            if vlm_plots:
                sections.append(
                    f"![VLM TTFT vs Input Tokens]({vlm_plots['ttft'].as_posix()})\n"
                )
                sections.append(
                    "*TTFT behaviour for multimodal prompts. Lower values mean faster first visual-token outputs.*\n\n"
                )
                sections.append(
                    f"![VLM Generation Time vs Output Tokens]({vlm_plots['tg'].as_posix()})\n"
                )
                sections.append(
                    "*Generation time vs output token count for multimodal responses. Lower values are faster.*\n"
                )

            # Add performance plots
            try:
                plot_vlm_performance(self.results_dir)
                sections.append("![VLM TTFT Performance](results/plots/vlm_ttft.png)\n")
                sections.append(
                    "*Time To First Token (TTFT) - Lower is better. "
                    "Measures response latency.*\n\n"
                )
                sections.append(
                    "![VLM Throughput Performance](results/plots/vlm_tps.png)\n"
                )
                sections.append(
                    "*Token Generation per second (TG) - Higher is better. "
                    "Measures token generation.*\n"
                )
            except Exception as e:
                print(f"⚠️  Failed to generate VLM plots: {e}")

        return "\n".join(sections)

    def _generate_footer(self) -> str:
        """Generate footer with notes.

        Returns:
            Footer section string
        """
        return (
            "\n---\n"
            "_All metrics are shown as median ± standard deviation across 3 runs. "
            "Lower times are better (faster performance)._\n"
        )
