"""README.md update utilities."""

from pathlib import Path


class ReadmeUpdater:
    """Update README.md with generated results section."""

    START_MARKER = "## Benchmark Results"
    END_MARKER = "\n## "

    def __init__(self, readme_path: Path = Path("README.md")):
        """Initialize with README path.

        Args:
            readme_path: Path to README.md file
        """
        self.readme_path = readme_path

    def update(self, results_section: str) -> None:
        """Update README with new results section.

        Args:
            results_section: Generated markdown results section
        """
        content = self._read_readme()
        new_content = self._replace_section(content, results_section)
        self._write_readme(new_content)
        print(f"✅ Updated {self.readme_path} with benchmark results")

    def _read_readme(self) -> str:
        """Read existing README content.

        Returns:
            README content or empty string if file doesn't exist
        """
        if self.readme_path.exists():
            return self.readme_path.read_text()
        return ""

    def _replace_section(self, content: str, new_section: str) -> str:
        """Replace existing results section or append.

        Args:
            content: Current README content
            new_section: New results section to insert

        Returns:
            Updated README content
        """
        if self.START_MARKER in content:
            # Replace existing section
            start_idx = content.find(self.START_MARKER)
            remaining = content[start_idx + len(self.START_MARKER) :]

            # Find next section
            end_idx = remaining.find(self.END_MARKER)
            if end_idx != -1:
                # Replace between markers
                return (
                    content[:start_idx] + new_section + "\n" + remaining[end_idx + 1 :]
                )
            else:
                # Replace to end of file
                return content[:start_idx] + new_section
        else:
            # Append to end
            return content.rstrip() + "\n\n" + new_section

    def _write_readme(self, content: str) -> None:
        """Write updated content to README.

        Args:
            content: Full README content to write
        """
        self.readme_path.write_text(content)
