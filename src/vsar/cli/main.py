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
    json_output: Annotated[
        bool, typer.Option("--json", help="Output results as JSON")
    ] = False,
    show_trace: Annotated[
        bool, typer.Option("--trace", help="Show trace DAG")
    ] = False,
    k: Annotated[
        Optional[int], typer.Option("--k", help="Number of results per query")
    ] = None,
) -> None:
    """Execute a VSAR program.

    Reads a .vsar file, executes all directives, inserts facts,
    and runs all queries.

    Example:
        vsar run program.vsar
        vsar run program.vsar --json
        vsar run program.vsar --trace
    """
    # Load program
    try:
        program = load_vsar(program_path)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error parsing program: {e}[/red]")
        raise typer.Exit(code=1)

    # Create engine
    try:
        engine = VSAREngine(program.directives)
    except Exception as e:
        console.print(f"[red]Error initializing engine: {e}[/red]")
        raise typer.Exit(code=1)

    # Insert facts
    for fact in program.facts:
        engine.insert_fact(fact)

    console.print(f"[green]Inserted {len(program.facts)} facts[/green]")

    # Execute queries
    if not program.queries:
        console.print("[yellow]No queries to execute[/yellow]")
        return

    results = []
    for query in program.queries:
        try:
            result = engine.query(query, k=k)
            results.append(result)
        except Exception as e:
            console.print(f"[red]Error executing query {query}: {e}[/red]")
            raise typer.Exit(code=1)

    # Format output
    if json_output:
        output = format_results_json(results)
        console.print(output)
    else:
        output = format_results_table(results)
        console.print(output)

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
    model_dim: Annotated[
        int, typer.Option("--dim", help="Model dimension")
    ] = 8192,
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

    directives = [
        Directive(name="model", params={"type": "FHRR", "dim": model_dim, "seed": seed})
    ]
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
    output = format_stats(stats)
    console.print(output)


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

    directives = [
        Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
    ]
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
    kb_path: Annotated[
        Optional[Path], typer.Argument(help="Path to knowledge base (HDF5)")
    ] = None,
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

    directives = [
        Directive(name="model", params={"type": "FHRR", "dim": 8192, "seed": 42})
    ]
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
    output = format_stats(stats)
    console.print(output)


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
    console.print("[bold cyan]VSAR Interactive REPL[/bold cyan]")
    console.print("Type 'help' for commands, 'exit' to quit\n")

    engine = None
    program = None

    while True:
        try:
            # Read input
            user_input = console.input("[bold green]> [/bold green]").strip()

            if not user_input:
                continue

            # Parse command
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()

            # Handle commands
            if command in ["exit", "quit"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif command == "help":
                console.print("\n[bold]Available Commands:[/bold]")
                console.print("  [cyan]load <file>[/cyan]        Load a VSAR program file")
                console.print("  [cyan]query <query>[/cyan]      Execute a query (e.g., parent(alice, X)?)")
                console.print("  [cyan]stats[/cyan]              Show knowledge base statistics")
                console.print("  [cyan]help[/cyan]               Show this help message")
                console.print("  [cyan]exit[/cyan] or [cyan]quit[/cyan]      Exit REPL\n")

            elif command == "load":
                if len(parts) < 2:
                    console.print("[red]Error: Usage: load <file>[/red]")
                    continue

                file_path = Path(parts[1])
                try:
                    program = load_vsar(file_path)
                    engine = VSAREngine(program.directives)

                    # Insert facts
                    for fact in program.facts:
                        engine.insert_fact(fact)

                    console.print(f"[green]Loaded {file_path}[/green]")
                    console.print(f"[green]Inserted {len(program.facts)} facts[/green]")

                except FileNotFoundError as e:
                    console.print(f"[red]{e}[/red]")
                except Exception as e:
                    console.print(f"[red]Error loading file: {e}[/red]")

            elif command == "query":
                if engine is None:
                    console.print("[red]Error: No program loaded. Use 'load <file>' first[/red]")
                    continue

                if len(parts) < 2:
                    console.print("[red]Error: Usage: query <query>[/red]")
                    console.print("[yellow]Example: query parent(alice, X)?[/yellow]")
                    continue

                query_text = parts[1].strip()

                # Parse query using the parser
                try:
                    from vsar.language.parser import parse

                    # Wrap in a minimal program to parse
                    program_text = f"@model FHRR(dim=8192, seed=42);\nquery {query_text}"
                    parsed_program = parse(program_text)

                    if not parsed_program.queries:
                        console.print("[red]Error: Invalid query syntax[/red]")
                        continue

                    query = parsed_program.queries[0]

                    # Execute query
                    result = engine.query(query, k=10)

                    # Display results
                    if result.results:
                        output = format_results_table([result])
                        console.print(output)
                    else:
                        console.print("[yellow]No results found[/yellow]")

                except Exception as e:
                    console.print(f"[red]Error executing query: {e}[/red]")

            elif command == "stats":
                if engine is None:
                    console.print("[red]Error: No program loaded. Use 'load <file>' first[/red]")
                    continue

                stats = engine.stats()
                output = format_stats(stats)
                console.print(output)

            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                console.print("[yellow]Type 'help' for available commands[/yellow]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            continue
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue


if __name__ == "__main__":
    app()
