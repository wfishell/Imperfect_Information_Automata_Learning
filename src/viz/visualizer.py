"""
Rich terminal visualizer for the L* + MCTS strategy learner.

Completely isolated from the algorithm — receives data objects and renders them.
Call from learner_viz.py; do not import from any algorithm module.
"""

from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.columns import Columns


class Visualizer:

    def __init__(self) -> None:
        self.console = Console()

    # ------------------------------------------------------------------
    # Game tree
    # ------------------------------------------------------------------

    def show_game_tree(self, root, max_depth: int = 4) -> None:
        self.console.print(Rule("[bold cyan]Game Tree[/bold cyan]"))
        tree = Tree(self._node_label(root, action=None))
        self._build_tree(root, tree, depth=0, max_depth=max_depth)
        self.console.print(tree)
        self.console.print()

    def _node_label(self, node, action) -> str:
        color = "blue" if node.player == "P1" else "green"
        label = f"[{color}]{node.player}[/{color}]  val=[yellow]{node.value}[/yellow]"
        if node.is_terminal():
            label += "  [red bold](terminal)[/red bold]"
        if action:
            label = f"[bold]{action}[/bold] → " + label
        return label

    def _build_tree(self, node, rich_node, depth, max_depth) -> None:
        if depth >= max_depth and not node.is_terminal():
            rich_node.add(f"[dim]… ({len(node.children)} children, depth {depth})[/dim]")
            return
        for action, child in node.children.items():
            child_node = rich_node.add(self._node_label(child, action))
            self._build_tree(child, child_node, depth + 1, max_depth)

    # ------------------------------------------------------------------
    # Round header
    # ------------------------------------------------------------------

    def show_round_header(self, round_num: int) -> None:
        self.console.print()
        self.console.print(Rule(f"[bold magenta]MCTS Round {round_num}[/bold magenta]"))

    # ------------------------------------------------------------------
    # L* hypothesis automaton
    # ------------------------------------------------------------------

    def show_hypothesis(self, model, p1_alphabet: list[str]) -> None:
        table = Table(
            title=f"Hypothesis Automaton  ({len(model.states)} states)",
            show_header=True,
            header_style="bold",
        )
        table.add_column("State", style="bold cyan", no_wrap=True)
        for inp in p1_alphabet:
            table.add_column(f"P1 = [bold]{inp}[/bold]", justify="center")

        for state in sorted(model.states, key=lambda s: s.state_id):
            is_init = (state == model.initial_state)
            state_label = f"s{state.state_id}{'  ◀ init' if is_init else ''}"
            row = [state_label]
            for inp in p1_alphabet:
                output = state.output_fun.get(inp, "?")
                nxt    = state.transitions.get(inp)
                nxt_id = f"s{nxt.state_id}" if nxt else "?"
                out_color = "green" if output not in ("?", None) else "dim"
                row.append(f"[{out_color}]{output}[/{out_color}] → {nxt_id}")
            table.add_row(*row)

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Table B
    # ------------------------------------------------------------------

    def show_table_b(self, table_b, top_n: int = 12) -> None:
        entries = []
        for trace_key, actions in table_b._nodes.items():
            for action, stats in actions.items():
                entries.append((list(trace_key), action, stats))

        if not entries:
            self.console.print("[dim]Table B is empty.[/dim]")
            return

        entries.sort(key=lambda x: x[2].visits, reverse=True)
        shown = entries[:top_n]

        total_nodes  = len(table_b._nodes)
        total_edges  = sum(len(v) for v in table_b._nodes.values())
        pruned_edges = sum(s.zero_prob for v in table_b._nodes.values() for s in v.values())

        table = Table(
            title=(f"Table B  —  top {len(shown)} of {total_edges} edges  "
                   f"({pruned_edges} pruned)"),
            show_header=True,
            header_style="bold",
        )
        table.add_column("Trace prefix", no_wrap=True)
        table.add_column("Action", justify="center")
        table.add_column("Visits", justify="right")
        table.add_column("Value",  justify="right")
        table.add_column("Pruned", justify="center")

        for trace, action, stats in shown:
            prefix_str = " → ".join(trace) if trace else "ε"
            val_color  = "green" if stats.value >= 0.5 else "red"
            pruned_str = "[red]✗[/red]" if stats.zero_prob else "[dim]–[/dim]"
            table.add_row(
                f"[dim]{prefix_str}[/dim]",
                f"[bold]{action}[/bold]",
                str(stats.visits),
                f"[{val_color}]{stats.value:.3f}[/{val_color}]",
                pruned_str,
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Deviation leaves
    # ------------------------------------------------------------------

    def show_deviations(self, deviation_leaves: dict) -> None:
        if not deviation_leaves:
            self.console.print("[dim]No deviation leaves recorded yet.[/dim]\n")
            return

        table = Table(
            title=f"Deviation Points  ({len(deviation_leaves)} tracked)",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Deviation point (trace)", no_wrap=True)
        table.add_column("Leaves explored", justify="right")
        table.add_column("Sample leaf", no_wrap=True)

        for dev_tuple, leaves in sorted(deviation_leaves.items(),
                                         key=lambda x: -len(x[1])):
            dev_str    = " → ".join(dev_tuple) if dev_tuple else "ε"
            sample     = " → ".join(leaves[-1]) if leaves else "–"
            table.add_row(
                f"[cyan]{dev_str}[/cyan]",
                str(len(leaves)),
                f"[dim]{sample}[/dim]",
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Improvement / convergence
    # ------------------------------------------------------------------

    def show_improvement(self, improvement) -> None:
        if improvement is None:
            self.console.print(
                Panel("[green bold]No improvement found — converged.[/green bold]",
                      border_style="green")
            )
        else:
            p1_seq = " → ".join(str(x) for x in improvement)
            self.console.print(
                Panel(
                    f"[yellow bold]Improvement found![/yellow bold]\n"
                    f"Counterexample P1 sequence: [bold]{p1_seq}[/bold]\n"
                    f"Restarting L* with updated strategy.",
                    border_style="yellow",
                )
            )
        self.console.print()

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def show_scores(self, scores: dict, round_num: int) -> None:
        norm  = scores["normalised"]
        color = "green" if norm >= 0.8 else ("yellow" if norm >= 0.4 else "red")

        table = Table(
            title=f"Strategy Quality — Round {round_num}",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Strategy", style="bold")
        table.add_column("Mean score", justify="right")

        table.add_row("Optimal",    f"{scores['optimal_mean']:.2f}")
        table.add_row("Learned",    f"{scores['learned_mean']:.2f}")
        table.add_row("Random",     f"{scores['random_mean']:.2f}")
        table.add_row(
            "Normalised",
            f"[{color}]{norm:.3f}[/{color}]  [dim](0 = random, 1 = optimal)[/dim]",
        )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    def show_final_summary(self, model, sul, eq_oracle, table_b) -> None:
        self.console.print(Rule("[bold green]Converged[/bold green]"))
        summary = (
            f"[bold]States[/bold]         : {len(model.states)}\n"
            f"[bold]Membership queries[/bold] : {sul.num_queries}\n"
            f"[bold]Equivalence queries[/bold]: {eq_oracle.num_queries}\n"
            f"[bold]{table_b.summary()}[/bold]"
        )
        self.console.print(Panel(summary, title="Final Summary", border_style="green"))
        self.console.print()
