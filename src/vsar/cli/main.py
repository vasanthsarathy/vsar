"""VSAR CLI application."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from typing_extensions import Annotated

from vsar.cli.formatters import (
    format_results_json,
    format_results_table,
    format_stats,
    format_trace_dag,
)
from vsar.language.loader import load_facts, load_vsar
from vsar.semantics.engine import VSAREngine

app = typer.Typer(
    name="vsar",
    help="VSAR - VSA-grounded Reasoning CLI",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    program_path: Annotated[Path, typer.Argument(help="Path to .vsar program file")],
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON")] = False,
    show_trace: Annotated[bool, typer.Option("--trace", help="Show trace DAG")] = False,
    k: Annotated[Optional[int], typer.Option("--k", help="Number of results per query")] = None,
) -> None:
    """Execute a VSAR program.

    Reads a .vsar file, executes all directives, inserts facts,
    and runs all queries.

    Example:
        vsar run program.vsar
        vsar run program.vsar --json
        vsar run program.vsar --trace
    """
    # Load program with status
    with console.status("[bold blue]Loading program...", spinner="dots"):
        try:
            program = load_vsar(program_path)
        except FileNotFoundError as e:
            console.print(f"[red][ERROR][/red] {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[red][ERROR][/red] Error parsing program: {e}")
            raise typer.Exit(code=1)

    console.print(f"[green][OK][/green] Loaded [cyan]{program_path.name}[/cyan]")

    # Create engine
    try:
        with console.status("[bold blue]Initializing engine...", spinner="dots"):
            engine = VSAREngine(program.directives)
    except Exception as e:
        console.print(f"[red][ERROR][/red] Error initializing engine: {e}")
        raise typer.Exit(code=1)

    # Insert facts with progress
    if program.facts:
        with console.status(f"[bold blue]Inserting {len(program.facts)} facts...", spinner="dots"):
            for fact in program.facts:
                engine.insert_fact(fact)
        console.print(f"[green][OK][/green] Inserted [cyan]{len(program.facts)}[/cyan] facts")

    # Show rules info if present
    if program.rules:
        console.print(f"[blue][INFO][/blue] Found [cyan]{len(program.rules)}[/cyan] rules")

    # Execute queries
    if not program.queries:
        console.print("[yellow][WARN][/yellow] No queries to execute")
        return

    console.print()  # Blank line before results

    results = []
    for i, query in enumerate(program.queries, 1):
        try:
            # Pass rules to query if program has rules
            with console.status(f"[bold blue]Executing query {i}/{len(program.queries)}...", spinner="dots"):
                result = engine.query(query, rules=program.rules if program.rules else None, k=k)
            results.append(result)
        except Exception as e:
            console.print(f"[red][ERROR][/red] Error executing query: {e}")
            raise typer.Exit(code=1)

    # Format output
    if json_output:
        output = format_results_json(results)
        console.print(output)
    else:
        format_results_table(results)

    # Show trace if requested
    if show_trace:
        trace_output = format_trace_dag(engine.trace)
        console.print(trace_output)


@app.command()
def ingest(
    facts_path: Annotated[Path, typer.Argument(help="Path to facts file")],
    kb_path: Annotated[
        Optional[Path],
        typer.Option("--kb", help="Path to save knowledge base (HDF5)"),
    ] = None,
    predicate: Annotated[
        Optional[str],
        typer.Option("--predicate", "-p", help="Predicate name for CSV files"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Format: auto, csv, jsonl, vsar"),
    ] = "auto",
    model_dim: Annotated[int, typer.Option("--dim", help="Model dimension")] = 8192,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
) -> None:
    """Ingest facts from CSV, JSONL, or VSAR file.

    Example:
        vsar ingest facts.csv --kb kb.h5
        vsar ingest facts.jsonl --format jsonl --kb kb.h5
        vsar ingest facts.csv --predicate parent --kb kb.h5
    """
    # Load facts
    try:
        facts = load_facts(facts_path, format=format, predicate=predicate)
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {facts_path}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading facts: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Loaded {len(facts)} facts[/green]")

    # Create engine with default config
    from vsar.language.ast import Directive

    directives = [Directive(name="model", params={"type": "FHRR", "dim": model_dim, "seed": seed})]
    engine = VSAREngine(directives)

    # Insert facts
    for fact in facts:
        engine.insert_fact(fact)

    console.print(f"[green]Inserted {len(facts)} facts[/green]")

    # Save KB if path provided
    if kb_path:
        engine.save_kb(kb_path)
        console.print(f"[green]Saved KB to {kb_path}[/green]")

    # Show stats
    stats = engine.stats()
    format_stats(stats)


@app.command()
def export(
    kb_path: Annotated[Path, typer.Argument(help="Path to knowledge base (HDF5)")],
    output_path: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    format: Annotated[
        str,
        typer.Option("--format", "-f", help="Format: json or jsonl"),
    ] = "json",
) -> None:
    """Export knowledge base to JSON or JSONL.

    Example:
        vsar export kb.h5 --format json
        vsar export kb.h5 --format jsonl --output facts.jsonl
    """
    # Create engine and load KB
    from vsar.language.ast import Directive

    directives = [Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})]
    engine = VSAREngine(directives)

    try:
        engine.load_kb(kb_path)
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {kb_path}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading KB: {e}[/red]")
        raise typer.Exit(code=1)

    # Export
    try:
        data = engine.export_kb(format)
    except Exception as e:
        console.print(f"[red]Error exporting KB: {e}[/red]")
        raise typer.Exit(code=1)

    # Write to file or stdout
    if output_path:
        output_path.write_text(data if isinstance(data, str) else str(data))
        console.print(f"[green]Exported to {output_path}[/green]")
    else:
        console.print(data)


@app.command()
def inspect(
    kb_path: Annotated[Optional[Path], typer.Argument(help="Path to knowledge base (HDF5)")] = None,
) -> None:
    """Inspect knowledge base statistics.

    Example:
        vsar inspect kb.h5
    """
    if not kb_path:
        console.print("[yellow]No KB path provided[/yellow]")
        return

    # Create engine and load KB
    from vsar.language.ast import Directive

    directives = [Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})]
    engine = VSAREngine(directives)

    try:
        engine.load_kb(kb_path)
    except FileNotFoundError:
        console.print(f"[red]Error: File not found: {kb_path}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error loading KB: {e}[/red]")
        raise typer.Exit(code=1)

    # Show stats
    stats = engine.stats()
    format_stats(stats)


@app.command()
def repl() -> None:
    """Start interactive REPL mode.

    Load VSAR files and execute queries interactively.

    Example:
        vsar repl
        > load family.vsar
        > query parent(alice, X)?
        > stats
        > exit
    """
    from rich.panel import Panel

    console.print(
        Panel(
            "[bold cyan]VSAR Interactive REPL[/bold cyan]\n"
            "Type [yellow]help[/yellow] for commands, [yellow]exit[/yellow] to quit",
            border_style="cyan",
        )
    )
    console.print()

    engine = None
    program = None

    while True:
        try:
            # Read input
            user_input = console.input("[bold cyan]vsar>[/bold cyan] ").strip()

            if not user_input:
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()

            # Handle commands
            if command in ["exit", "quit"]:
                console.print("[bold green]Goodbye![/bold green]")
                break

            elif command == "help":
                console.print()
                help_panel = Panel(
                    "[bold]Available Commands:[/bold]\n\n"
                    "  [cyan]load[/cyan] [yellow]<file>[/yellow]      Load a VSAR program file\n"
                    "  [cyan]query[/cyan] [yellow]<query>[/yellow]    Execute a query (e.g., parent(alice, X)?)\n"
                    "  [cyan]stats[/cyan]             Show knowledge base statistics\n"
                    "  [cyan]help[/cyan]              Show this help message\n"
                    "  [cyan]exit[/cyan] / [cyan]quit[/cyan]        Exit REPL",
                    title="[bold]Help[/bold]",
                    border_style="blue",
                )
                console.print(help_panel)
                console.print()

            elif command == "load":
                if len(parts) < 2:
                    console.print("[red][ERROR][/red] Usage: [cyan]load[/cyan] [yellow]<file>[/yellow]")
                    continue

                file_path = Path(parts[1])
                try:
                    with console.status("[blue]Loading program...", spinner="dots"):
                        program = load_vsar(file_path)
                        engine = VSAREngine(program.directives)

                        # Insert facts
                        for fact in program.facts:
                            engine.insert_fact(fact)

                    console.print(f"[green][OK][/green] Loaded [cyan]{file_path}[/cyan]")
                    console.print(f"[green][OK][/green] Inserted [cyan]{len(program.facts)}[/cyan] facts")

                    if program.rules:
                        console.print(f"[blue][INFO][/blue] Found [cyan]{len(program.rules)}[/cyan] rules")

                except FileNotFoundError as e:
                    console.print(f"[red][ERROR][/red] {e}")
                except Exception as e:
                    console.print(f"[red][ERROR][/red] Error loading file: {e}")

            elif command == "query":
                if engine is None:
                    console.print("[red][ERROR][/red] No program loaded. Use [cyan]load <file>[/cyan] first")
                    continue

                if len(parts) < 2:
                    console.print("[red][ERROR][/red] Usage: [cyan]query[/cyan] [yellow]<query>[/yellow]")
                    console.print("[dim]Example: query parent(alice, X)?[/dim]")
                    continue

                query_text = parts[1].strip()

                # Parse query using the parser
                try:
                    from vsar.language.parser import parse

                    # Wrap in a minimal program to parse
                    program_text = f"@model FHRR(dim=8192, seed=42);\nquery {query_text}"
                    parsed_program = parse(program_text)

                    if not parsed_program.queries:
                        console.print("[red][ERROR][/red] Invalid query syntax")
                        continue

                    query = parsed_program.queries[0]

                    # Execute query with rules if available
                    with console.status("[blue]Executing query...", spinner="dots"):
                        result = engine.query(
                            query,
                            rules=program.rules if program and program.rules else None,
                            k=10
                        )

                    # Display results
                    console.print()
                    if result.results:
                        format_results_table([result])
                    else:
                        console.print("[yellow][WARN][/yellow] No results found")
                    console.print()

                except Exception as e:
                    console.print(f"[red][ERROR][/red] Error executing query: {e}")

            elif command == "stats":
                if engine is None:
                    console.print("[red][ERROR][/red] No program loaded. Use [cyan]load <file>[/cyan] first")
                    continue

                console.print()
                stats = engine.stats()
                format_stats(stats)
                console.print()

            else:
                console.print(f"[red][ERROR][/red] Unknown command: [yellow]{command}[/yellow]")
                console.print("[dim]Type [cyan]help[/cyan] for available commands[/dim]")

        except KeyboardInterrupt:
            console.print("\n[yellow][WARN][/yellow] Use [cyan]exit[/cyan] to quit")
            continue
        except EOFError:
            console.print("\n[bold green]Goodbye![/bold green]")
            break
        except Exception as e:
            console.print(f"[red][ERROR][/red] Error: {e}")
            continue


if __name__ == "__main__":
    app()
