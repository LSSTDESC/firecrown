"""Classes for tracking parameter usage by Updatable objects."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class UpdatableUsageRecord:
    """Dataclass to record the usage of parameters by an Updatable object."""

    cls: str
    prefix: str | None
    obj_id: int
    sampler_params: list[str]
    internal_params: list[str]
    child_records: list[UpdatableUsageRecord]
    already_updated: bool = False

    @property
    def is_empty(self) -> bool:
        """Check if the record is empty.

        Return True if the record has no sampler parameters, internal parameters, or
        child records.
        """
        if self.sampler_params:
            return False
        if self.internal_params:
            return False
        return all(cr.already_updated for cr in self.child_records)

    @property
    def is_empty_parent(self) -> bool:
        """Check if the record is an empty parent.

        Return True if the record has no sampler or internal parameters and exactly one
        child record.
        """
        return (
            (len(self.sampler_params) == 0)
            and (len(self.internal_params) == 0)
            and (len(self.child_records) == 1)
        )

    def get_log_lines(
        self, level: int = 0, parent: str | None = None, print_empty: bool = False
    ) -> list[str]:
        """Print the usage record."""
        fullname = (
            f"{self.cls}({self.prefix})" if self.prefix is not None else f"{self.cls}"
        )
        if parent is not None:
            fullname_with_parent = f"{parent} => {fullname}"
        else:
            fullname_with_parent = fullname

        if self.is_empty_parent:
            return self.child_records[0].get_log_lines(
                level, fullname_with_parent, print_empty=print_empty
            )

        if self.is_empty and (not print_empty):
            return []

        lines = []
        next_level = level + 2
        indent = " " * level
        indent_next = " " * (next_level)
        if self.already_updated:
            lines.append(f"{indent}{fullname_with_parent}: (already updated)")
            return lines
        lines.append(f"{indent}{fullname_with_parent}: ")
        if self.sampler_params:
            lines.append(
                f"{indent_next}Sampler parameters used:  {self.sampler_params}"
            )
        if self.internal_params:
            lines.append(
                f"{indent_next}Internal parameters used: {self.internal_params}"
            )
        for child in self.child_records:
            lines += child.get_log_lines(next_level, print_empty=print_empty)

        return lines
