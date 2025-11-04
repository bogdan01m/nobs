"""Base class for markdown table generators."""

from abc import ABC, abstractmethod
from typing import Any


class BaseTableGenerator(ABC):
    """Abstract base class for markdown table generators."""

    def __init__(self, results: list[dict[str, Any]]):
        """Initialize with benchmark results.

        Args:
            results: List of benchmark result dictionaries
        """
        self.results = results

    @abstractmethod
    def generate(self) -> str:
        """Generate markdown table.

        Returns:
            Markdown table string, or empty string if no data
        """
        pass

    def _has_data(self) -> bool:
        """Check if there's data to generate.

        Returns:
            True if results list is not empty
        """
        return bool(self.results)

    def _build_header(self, columns: list[str]) -> list[str]:
        """Build markdown table header.

        Args:
            columns: List of column names

        Returns:
            List of two strings: header row and separator row
        """
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join(["------"] * len(columns)) + "|"
        return [header, separator]
