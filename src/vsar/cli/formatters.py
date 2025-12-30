"""Output formatters for VSAR CLI."""

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from vsar.semantics.engine import QueryResult
from vsar.trace.collector import TraceCollector


def format_results_table(results: list[QueryResult]) -> str:
    """Format query results as a rich table.

    Args:
        results: List of query results

    Returns:
        Formatted table string

    Example:
        >>> result = QueryResult(...)
        >>> output = format_results_table([result])
    """
    console = Console()

    for i, result in enumerate(results):
        if i > 0:
            console.print()  # Blank line between queries

        # Create table for this query
        table = Table(title=f"Query: {_format_query(result.query)}")
        table.add_column("Entity", style="cyan")
        table.add_column("Score", style="green")

        # Add rows
        for entity, score in result.results:
            table.add_row(entity, f"{score:.4f}")

        # Capture output
        with console.capture() as capture:
            console.print(table)

        return capture.get()

    return ""


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
        data.append({
            "query": {
                "predicate": result.query.predicate,
                "args": result.query.args,
            },
            "results": [
                {"entity": entity, "score": score}
                for entity, score in result.results
            ],
            "trace_id": result.trace_id,
        })

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


def format_stats(stats: dict[str, Any]) -> str:
    """Format KB statistics as a table.

    Args:
        stats: Statistics dictionary

    Returns:
        Formatted table string

    Example:
        >>> stats = {"total_facts": 100, "predicates": {...}}
        >>> output = format_stats(stats)
    """
    console = Console()

    # Create summary table
    summary = Table(title="Knowledge Base Statistics")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Total Facts", str(stats["total_facts"]))
    summary.add_row("Predicates", str(len(stats["predicates"])))

    # Create predicate details table
    details = Table(title="Predicate Details")
    details.add_column("Predicate", style="yellow")
    details.add_column("Fact Count", style="green")

    for predicate, count in sorted(stats["predicates"].items()):
        details.add_row(predicate, str(count))

    # Capture output
    with console.capture() as capture:
        console.print(summary)
        console.print()
        console.print(details)

    return capture.get()


def _format_query(query: Any) -> str:
    """Format a query for display.

    Args:
        query: Query object

    Returns:
        Formatted query string
    """
    args_str = ", ".join(
        arg if arg is not None else "X"
        for arg in query.args
    )
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
