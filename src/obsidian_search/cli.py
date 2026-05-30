"""CLI entry point — grows phase by phase."""

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from obsidian_search.config import settings
from obsidian_search.parser import parse_vault
from obsidian_search.chunker import chunk_note

app = typer.Typer(help="Obsidian Semantic Search — learning project")


@app.command()
def parse(
    folder: str = typer.Option(None, help="Vault subfolder (e.g. 05-Knowledge)"),
    limit: int = typer.Option(5, help="Number of notes to show"),
    output_json: bool = typer.Option(False, "--json", help="Print full JSON"),
):
    """Phase 1 — parse vault notes and inspect structure."""
    folders = [folder] if folder else None
    notes = parse_vault(settings.vault_path, folders=folders)

    if output_json:
        data = [n.to_dict() for n in notes[:limit]]
        rprint(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return

    table = Table(title=f"Parsed notes ({len(notes)} total)", show_lines=True)
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Area", style="green")
    table.add_column("Tags", max_width=30)
    table.add_column("Headings", max_width=30)
    table.add_column("Hash")

    for note in notes[:limit]:
        table.add_row(
            note.title,
            note.area,
            ", ".join(note.tags[:4]),
            " > ".join(note.headings[:3]),
            note.content_hash,
        )

    rprint(table)


@app.command()
def chunks(
    folder: str = typer.Option("05-Knowledge", help="Vault subfolder"),
    strategy: str = typer.Option("section", help="'section' or 'sliding'"),
    limit: int = typer.Option(3, help="Number of notes to chunk"),
):
    """Phase 3 — inspect chunking output for a few notes."""
    notes = parse_vault(settings.vault_path, folders=[folder])

    for note in notes[:limit]:
        ch = chunk_note(note, strategy=strategy,
                        max_tokens=settings.chunk_max_tokens,
                        overlap=settings.chunk_overlap_tokens)
        rprint(f"\n[bold cyan]{note.title}[/bold cyan] — {len(ch)} chunks ({strategy})")
        for c in ch[:2]:
            rprint(f"  [yellow]{c.chunk_id}[/yellow] | {c.token_count} tokens | heading: {c.heading_path!r}")
            rprint(f"  [dim]{c.text[:120].replace(chr(10), ' ')}...[/dim]")


if __name__ == "__main__":
    app()
