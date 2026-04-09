from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IDENTIFIER_FIELDS = {"code", "day", "pubDate", "statDate"}
DISALLOWED_FIELDS = {"paused"}


@dataclass(frozen=True)
class TableSpec:
    name: str
    fields: dict[str, str]


@dataclass(frozen=True)
class DataDictionary:
    tables: dict[str, TableSpec]
    allowed_factor_fields: frozenset[str]

    def has_field(self, field_name: str) -> bool:
        return field_name in self.allowed_factor_fields


def load_data_dictionary(path: str | Path) -> DataDictionary:
    markdown = Path(path).read_text(encoding="utf-8")
    tables: dict[str, TableSpec] = {}
    current_table: str | None = None
    current_fields: dict[str, str] = {}
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("### 表名:"):
            if current_table is not None:
                tables[current_table] = TableSpec(name=current_table, fields=current_fields)
            current_table = line.split(":", 1)[1].strip()
            current_fields = {}
            continue
        if not current_table or not line.startswith("|"):
            continue
        columns = [part.strip() for part in line.strip("|").split("|")]
        if len(columns) < 2:
            continue
        field_name, description = columns[0], columns[1]
        if field_name in {"列名", "------"} or not field_name:
            continue
        current_fields[field_name] = description
    if current_table is not None:
        tables[current_table] = TableSpec(name=current_table, fields=current_fields)

    allowed_fields: set[str] = set()
    for table in tables.values():
        for field_name in table.fields:
            if field_name in IDENTIFIER_FIELDS or field_name in DISALLOWED_FIELDS:
                continue
            allowed_fields.add(field_name)
    return DataDictionary(tables=tables, allowed_factor_fields=frozenset(sorted(allowed_fields)))
