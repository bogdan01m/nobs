"""Generate markdown tables from benchmark results JSON files.

This module has been refactored. The main logic is now in src/reporting/.
This file serves as a simple entry point for backward compatibility.
"""

from pathlib import Path

import seaborn as sns

from reporting.markdown.sections import ResultsSectionGenerator
from reporting.readme_updater import ReadmeUpdater

sns.set_style("darkgrid")


def update_readme(
    results_dir: Path = Path("results"), readme_path: Path = Path("README.md")
) -> None:
    """Update README.md with summary table only.

    Args:
        results_dir: Directory containing benchmark result JSON files
        readme_path: Path to README.md file
    """
    generator = ResultsSectionGenerator(results_dir)
    results_section = generator.generate(summary_only=True)

    updater = ReadmeUpdater(readme_path)
    updater.update(results_section)


def generate_results_docs(
    results_dir: Path = Path("results"), output_path: Path = Path("docs/results.md")
) -> None:
    """Generate docs/results.md with full results including plots.

    Preserves any custom content before the "## Benchmark Results" marker.

    Args:
        results_dir: Directory containing benchmark result JSON files
        output_path: Path to output markdown file
    """
    # Use symlinked plots directory for MkDocs
    generator = ResultsSectionGenerator(results_dir, plot_base_path="")
    results_section = generator.generate(include_plots=True)

    # Ensure docs directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve custom content before the marker
    custom_header = ""
    marker = "## Benchmark Results"

    if output_path.exists():
        existing_content = output_path.read_text()
        if marker in existing_content:
            # Keep everything before the marker
            custom_header = existing_content.split(marker, 1)[0]

    # Combine custom header with generated results
    final_content = custom_header + results_section
    output_path.write_text(final_content)
    print(f"✅ Generated {output_path} with benchmark results")


if __name__ == "__main__":
    update_readme()
    generate_results_docs()
