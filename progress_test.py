from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.console import Console
from rich.theme import Theme
import time

def simulate_work(steps):
    time.sleep(0.1)  # Simulate some work

# Create console with custom theme
console = Console(theme=Theme({
    "info": "cyan",
    "warning": "yellow",
    "danger": "red",
    "success": "green"
}))

# Style 1: Professional with Spinner
console.print("\n[bold cyan]Style 1: Professional with Spinner[/]")
progress1 = Progress(
    SpinnerColumn(spinner_name="dots12"),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(complete_style="cyan", finished_style="cyan"),
    MofNCompleteColumn(),
    TimeRemainingColumn()
)

# Style 2: Detailed Progress
console.print("\n[bold cyan]Style 2: Detailed Progress[/]")
progress2 = Progress(
    TextColumn("[bold]{task.description}"),
    BarColumn(complete_style="green", finished_style="green"),
    TimeElapsedColumn(),
    "|",
    TimeRemainingColumn(),
    "|",
    MofNCompleteColumn()
)

# Style 3: Minimal Clean
console.print("\n[bold cyan]Style 3: Minimal Clean[/]")
progress3 = Progress(
    TextColumn("[bold]{task.description}"),
    BarColumn(complete_style="blue", finished_style="blue"),
    MofNCompleteColumn()
)

# Demonstrate all styles
total_steps = 50

# Style 1 Demo
with progress1:
    task1 = progress1.add_task("[cyan]Processing Data...", total=total_steps)
    for i in range(total_steps):
        simulate_work(i)
        progress1.update(task1, advance=1)

# Style 2 Demo
with progress2:
    task2 = progress2.add_task("[green]Analyzing Results...", total=total_steps)
    for i in range(total_steps):
        simulate_work(i)
        progress2.update(task2, advance=1)

# Style 3 Demo
with progress3:
    task3 = progress3.add_task("[blue]Finalizing...", total=total_steps)
    for i in range(total_steps):
        simulate_work(i)
        progress3.update(task3, advance=1)

console.print("\n[bold green]All progress styles demonstrated successfully![/]") 