"""Terminal tables for ``verify`` and ``compare`` (rich).

The plain-text reports (``format_check``, ``format_comparison``, ``format_combined``
in ``verify.py``) stay the reference: they are what logs, CI and tests read. This
module draws the same facts as tables. A coloured cell always carries its text
too (``met``, ``NOT met``, ``seen``, ``unseen``), so nothing depends on colour.
An axis that has no ``tables()`` is printed with its ``lines()``.
"""

from __future__ import annotations

import sys

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .report import MAX_FINDING_LINES, MAX_HINT_LINES
from .verifier import BAD, GOOD, ReportTable, pct, rate
from .verify import CheckReport, CombinedComparison, RolloutComparison, _short_node

SEVERITY_STYLE = {"ERROR": "bold red", "VIOLATION": "red", "WARNING": "yellow", "INFO": "dim", "NOTE": "yellow"}
MAX_NODES = 3
NON_TTY_WIDTH = 140


def make_console(file=None) -> Console:
    """A console that uses the terminal's width, or a wide fixed one when output is redirected."""
    stream = file or sys.stdout
    return Console(file=stream, width=None if stream.isatty() else NON_TTY_WIDTH, highlight=False)


def use_tables(mode: str, file=None) -> bool:
    """``auto``: tables on a terminal, plain text when redirected; ``table`` and ``text`` force one."""
    if mode == "auto":
        return (file or sys.stdout).isatty()
    return mode == "table"


# ---- Pieces ---------------------------------------------------------------------------


def _cell(value: str) -> Text:
    if value in GOOD:
        return Text(value, style="green")
    if value in BAD:
        return Text(value, style="bold red")
    return Text(value)


def _header(rows: list[tuple[str, str]], title: str) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for key, value in rows:
        grid.add_row(Text(key), Text(value))
    return Panel(grid, title=title, title_align="left", box=box.ROUNDED, expand=False)


def _table(spec: ReportTable) -> Group:
    table = Table(title=spec.title, title_justify="left", box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False)
    for i, column in enumerate(spec.columns):
        table.add_column(column, justify="left" if i == 0 else "right", overflow="fold")
    for row in spec.rows:
        table.add_row(*[_cell(v) for v in row])
    return Group(table, *[Text(f"  {note}", style="dim") for note in spec.notes])


def _axes(axes) -> list:
    out: list = []
    for axis in axes:
        tables = axis.tables()
        if tables:
            out += [_table(t) for t in tables]
        else:
            out += [Text(line) for line in axis.lines()]
    return out


def _findings(rows: list[tuple[str, str, str]]) -> Table | None:
    if not rows:
        return None
    table = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False, title="Findings", title_justify="left")
    table.add_column("Severity", no_wrap=True)
    table.add_column("Message", overflow="fold", ratio=3)
    table.add_column("Nodes", overflow="fold", ratio=1)
    for severity, message, nodes in rows:
        table.add_row(Text(severity, style=SEVERITY_STYLE.get(severity, "")), Text(message), Text(nodes, style="dim"))
    return table


def _capped(rows: list[tuple[str, str, str]], what: str) -> list[tuple[str, str, str]]:
    if len(rows) <= MAX_FINDING_LINES:
        return rows
    return rows[:MAX_FINDING_LINES] + [("", f"... and {len(rows) - MAX_FINDING_LINES} more {what} (all of them are in findings/*.ttl in the store)", "")]


def _nodes(finding) -> str:
    nodes = ", ".join(_short_node(n) for n in finding.focus_nodes[:MAX_NODES])
    more = f" (+{finding.count - MAX_NODES} more)" if finding.count > MAX_NODES else ""
    return nodes + more


def _result(text: str, good: bool) -> Text:
    return Text(f"RESULT: {text}", style="bold green" if good else "bold red")


# ---- Reports --------------------------------------------------------------------------


def render_check(report: CheckReport, console: Console) -> None:
    console.print(_header([("Store", report.store), ("Spec", report.spec + (f"  ({report.spec_path})" if report.spec_path else "")), ("Episodes", str(report.num_episodes))], "Dataset check"))
    for part in _axes(report.axes):
        console.print(part)
    rows = _capped([("ERROR", e, "") for e in report.errors], "ERROR lines")
    for kind, findings in (("VIOLATION", report.violations), ("WARNING", report.warnings)):
        by_verifier: dict[str, list[tuple[str, str, str]]] = {}
        for f in findings:
            by_verifier.setdefault(f.verifier, []).append((kind, f.message, _nodes(f)))
        for items in by_verifier.values():
            rows += _capped(items, f"{kind} lines")
    table = _findings(rows)
    if table is not None:
        console.print(table)
    console.print(_result("coverage spec met" if report.passed else "coverage spec NOT met", report.passed))


def render_comparison(result: RolloutComparison, console: Console, hints: bool = True) -> None:
    aborted = f", {result.aborted} aborted (not counted)" if result.aborted else ""
    rollout = result.rollout + (f"  ({result.eval_file}, seed {result.seed})" if result.eval_file else "")
    console.print(
        _header(
            [("Store", result.store), ("Spec", result.spec), ("Rollout", rollout), ("Runs", f"{result.runs} finished, {result.successes} succeeded{aborted}")],
            "Rollout comparison",
        )
    )
    for part in _axes(result.axes):
        console.print(part)
    if result.axes and result.runs and result.passed:
        out = result.runs - result.runs_in_distribution
        table = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False, title=f"In distribution (covered on every axis: {', '.join(a.axis for a in result.axes)})", title_justify="left")
        for column in ("Runs in distribution", "Success in distribution", "Success outside"):
            table.add_column(column, justify="right")
        table.add_row(
            f"{result.runs_in_distribution}/{result.runs} ({pct(result.runs_in_distribution, result.runs)})",
            rate(result.successes_in_distribution, result.runs_in_distribution),
            rate(result.successes - result.successes_in_distribution, out),
        )
        console.print(table)
    if hints:
        lines = result.hint_lines()
        shown = lines[:MAX_HINT_LINES]
        if len(lines) > MAX_HINT_LINES:
            shown.append(f"... and {len(lines) - MAX_HINT_LINES} more hint lines (all of them are in findings/*.ttl in the store)")
        while shown and not shown[-1].strip():
            shown.pop()
        if shown:
            console.print(Panel(Text("\n".join(shown)), title="Suggested next steps", title_align="left", box=box.ROUNDED, expand=False))
    rows = _capped([("ERROR", e, "") for e in result.errors], "ERROR lines")
    rows += [(f.severity.upper(), f.message, _nodes(f)) for f in result.warnings]
    table = _findings(rows)
    if table is not None:
        console.print(table)
    if not result.passed:
        console.print(_result("comparison could not be made", False))
    elif result.fully_covered:
        console.print(_result("the dataset covers every rollout run", True))
    else:
        console.print(_result("the dataset does NOT cover every rollout run; see above", False))


def render_combined(result: CombinedComparison, console: Console) -> None:
    console.print(_header([("Rollout", result.rollout), ("Specs", ", ".join(result.specs) or "none")], "Combined (all spec files together)"))
    if result.skipped:
        console.print(
            Text(
                f"NOTE: not combined (comparison failed or has no axis): {', '.join(result.skipped)}; "
                "fix those specs to get a verdict over every axis",
                style="yellow",
            )
        )
    if result.specs and result.runs:
        out = result.runs - result.runs_in_distribution
        table = Table(box=box.SIMPLE_HEAD, header_style="bold", pad_edge=False, title=f"In distribution on every axis ({', '.join(result.axes)})", title_justify="left")
        for column in ("Runs in distribution", "Success in distribution", "Success outside"):
            table.add_column(column, justify="right")
        table.add_row(
            f"{result.runs_in_distribution}/{result.runs} ({pct(result.runs_in_distribution, result.runs)})",
            rate(result.successes_in_distribution, result.runs_in_distribution),
            rate(result.successes - result.successes_in_distribution, out),
        )
        console.print(table)
    if not result.specs:
        console.print(_result("no spec could be compared; nothing to combine", False))
    elif result.fully_covered:
        console.print(_result("the dataset covers every rollout run on every axis", True))
    else:
        console.print(_result("the dataset does NOT cover every rollout run on every axis" + (" given" if result.skipped else ""), False))
