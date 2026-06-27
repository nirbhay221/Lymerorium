#!/usr/bin/env python3
"""SwarmCore CLI - run a debate from the terminal."""


import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import time
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

import agents as ag
import knowledge_graph as kg
from api import run_simulation
from config import MAX_ROUNDS

console = Console()

STYLE_MAP = {
    "challenging": "red",
    "expansive": "cyan",
    "grounded": "green",
    "principled": "yellow",
    "technical": "blue",
    "analytical": "magenta",
    "provocative": "bright_red",
    "integrative": "bright_cyan",
}


def run(topic: str, max_rounds: int = MAX_ROUNDS) -> None:
    console.print(Panel.fit(
        "[bold cyan]SwarmCore[/bold cyan]\n[dim]Progressive Tooled Debate - Gemma 4 E2B - LangGraph[/dim]",
        border_style="cyan",
    ))
    console.print(f"\n[bold]Topic:[/bold] {topic}")
    console.print(f"[dim]Agents: {len(ag.SWARM)} - Rounds: {max_rounds}[/dim]\n")

    t0 = time.time()
    result = run_simulation(topic, max_rounds=max_rounds)
    elapsed = round(time.time() - t0, 1)

    if result.get("cache_hit"):
        console.print(f"[green]Cache hit[/green] - served from memory ({result.get('cache_age_hours', '?')}h ago)")
    else:
        console.print(Rule("[dim]Debate transcript[/dim]", style="dim"))
        agent_idx = ag.AGENT_INDEX
        for msg in result.get("messages", []):
            name = msg["agent"]
            style = STYLE_MAP.get(agent_idx.get(name, {}).get("style", ""), "white")
            console.print(f"\n[bold {style}][{name}][/bold {style}] Round {msg['round']}")
            console.print(msg["content"])
            time.sleep(0.05)

    console.print(Rule("[bold cyan]Verdict[/bold cyan]", style="cyan"))
    console.print(Panel(
        result.get("verdict", "(no verdict)"),
        border_style="cyan",
        title="[bold]Synthesizer[/bold]",
    ))

    path = result.get("report_path", "")
    if path:
        console.print(f"\n[dim]Report saved: {path}[/dim]")
    console.print(f"[dim]Completed in {elapsed}s - convergence={result.get('convergence_score', 0):.2f}[/dim]")

    kg.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[yellow]Usage:[/yellow] python main.py \"<topic>\" [rounds]")
        console.print("[dim]Example: python main.py \"Will edge AI replace cloud AI?\" 1[/dim]")
        sys.exit(1)

    topic_arg = sys.argv[1]
    rounds_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(topic_arg, rounds_arg)
