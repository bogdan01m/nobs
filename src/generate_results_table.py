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
    """Update README.md with generated results section.

    Args:
        results_dir: Directory containing benchmark result JSON files
        readme_path: Path to README.md file
    """
    generator = ResultsSectionGenerator(results_dir)
    results_section = generator.generate()

    updater = ReadmeUpdater(readme_path)
    updater.update(results_section)


if __name__ == "__main__":
    update_readme()
