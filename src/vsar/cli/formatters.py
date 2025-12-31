"""Output formatters for VSAR CLI."""

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from vsar.semantics.engine import QueryResult
from vsar.trace.collector import TraceCollector


def format_results_table(results: list[QueryResult]) -> None:
    """Format and print query results as rich tables.

    Args:
        results: List of query results

    Example:
        >>> result = QueryResult(...)
        >>> format_results_table([result])
    """
    console = Console()

    for i, result in enumerate(results):
        if i > 0:
            console.print()  # Blank line between queries

        # Create table for this query
        table = Table(
            title=f"[bold cyan]Query:[/bold cyan] [yellow]{_format_query(result.query)}[/yellow]",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
        )
        table.add_column("Entity", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right", style="green")

        # Add rows
        if not result.results:
            table.add_row("[dim]No results[/dim]", "[dim]-[/dim]")
        else:
            for entity, score in result.results:
                # Color code scores
                if score >= 0.9:
                    score_style = "bold green"
                elif score >= 0.7:
                    score_style = "green"
                elif score >= 0.5:
                    score_style = "yellow"
                else:
                    score_style = "red"

                table.add_row(entity, f"[{score_style}]{score:.4f}[/{score_style}]")

        console.print(table)


def format_results_json(results: list[QueryResult]) -> str:
    """Format query results as JSON.

    Args:
        results: List of query results

    Returns:
        JSON string

    Example:
        >>> result = QueryResult(...)
        >>> output = format_results_json([result])
    """
    data = []
    for result in results:
        data.append(
            {
                "query": {
                    "predicate": result.query.predicate,
                    "args": result.query.args,
                },
                "results": [{"entity": entity, "score": score} for entity, score in result.results],
                "trace_id": result.trace_id,
            }
        )

    return json.dumps(data, indent=2)


def format_trace_dag(trace: TraceCollector, event_id: str | None = None) -> str:
    """Format trace DAG as text.

    Args:
        trace: Trace collector
        event_id: Optional event ID to show subgraph for

    Returns:
        Formatted trace string

    Example:
        >>> collector = TraceCollector()
        >>> output = format_trace_dag(collector)
    """
    console = Console()

    # Get events to display
    if event_id:
        events = trace.get_subgraph(event_id)
    else:
        events = trace.get_dag()

    # Create table
    table = Table(title="Trace DAG")
    table.add_column("Event ID", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Payload", style="white")
    table.add_column("Parents", style="magenta")

    # Add rows
    for event in events:
        event_id_short = event.id[:8] + "..."
        payload_str = _format_payload(event.payload)
        parents_str = ", ".join(p[:8] + "..." for p in event.parent_ids) or "-"

        table.add_row(
            event_id_short,
            event.type,
            payload_str,
            parents_str,
        )

    # Capture output
    with console.capture() as capture:
        console.print(table)

    return capture.get()


def format_stats(stats: dict[str, Any]) -> None:
    """Format and print KB statistics as tables.

    Args:
        stats: Statistics dictionary

    Example:
        >>> stats = {"total_facts": 100, "predicates": {...}}
        >>> format_stats(stats)
    """
    console = Console()

    # Create summary table
    summary = Table(
        title="[bold cyan]Knowledge Base Statistics[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    summary.add_column("Metric", style="cyan", no_wrap=True)
    summary.add_column("Value", justify="right", style="green")

    summary.add_row("Total Facts", f"[bold]{stats['total_facts']}[/bold]")
    summary.add_row("Predicates", f"[bold]{len(stats['predicates'])}[/bold]")

    console.print(summary)

    # Create predicate details table if there are predicates
    if stats["predicates"]:
        console.print()
        details = Table(
            title="[bold cyan]Predicate Details[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
        )
        details.add_column("Predicate", style="yellow", no_wrap=True)
        details.add_column("Facts", justify="right", style="green")

        for predicate, count in sorted(stats["predicates"].items()):
            details.add_row(predicate, str(count))

        console.print(details)


def _format_query(query: Any) -> str:
    """Format a query for display.

    Args:
        query: Query object

    Returns:
        Formatted query string
    """
    args_str = ", ".join(arg if arg is not None else "X" for arg in query.args)
    return f"{query.predicate}({args_str})"


def _format_payload(payload: dict[str, Any]) -> str:
    """Format event payload for display.

    Args:
        payload: Payload dictionary

    Returns:
        Formatted payload string (truncated if needed)
    """
    # Truncate long payloads
    payload_str = str(payload)
    if len(payload_str) > 50:
        payload_str = payload_str[:47] + "..."
    return payload_str
