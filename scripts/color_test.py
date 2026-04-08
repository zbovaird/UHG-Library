from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Standard Colors
STANDARD_COLORS = [
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
]

# Bright Colors
BRIGHT_COLORS = [
    "bright_black", "bright_red", "bright_green", "bright_yellow",
    "bright_blue", "bright_magenta", "bright_cyan", "bright_white"
]

# RGB Colors Examples
RGB_COLORS = [
    "rgb(255,0,0)", "rgb(0,255,0)", "rgb(0,0,255)",
    "rgb(255,255,0)", "rgb(255,0,255)", "rgb(0,255,255)"
]

# Hex Colors Examples
HEX_COLORS = [
    "#ff0000", "#00ff00", "#0000ff",
    "#ffff00", "#ff00ff", "#00ffff"
]

def show_color_group(title, colors):
    console.print(f"\n[bold white]{title}[/]")
    table = Table(show_header=False, box=None)
    
    for color in colors:
        table.add_row(
            f"[{color}]■[/]",
            f"[{color}]Text in {color}[/]",
            f"[bold {color}]Bold text in {color}[/]"
        )
    
    console.print(table)

# Display all color groups
console.print(Panel.fit(
    "[bold]Rich Color Demonstration[/]",
    border_style="cyan"
))

show_color_group("Standard Colors", STANDARD_COLORS)
show_color_group("Bright Colors", BRIGHT_COLORS)
show_color_group("RGB Color Examples", RGB_COLORS)
show_color_group("Hex Color Examples", HEX_COLORS)

# Show color combinations
console.print("\n[bold white]Color Combination Examples[/]")
console.print("[red on white]Red on White Background[/]")
console.print("[black on yellow]Black on Yellow Background[/]")
console.print("[white on blue]White on Blue Background[/]")
console.print("[yellow on red]Yellow on Red Background[/]")

# Show style combinations
console.print("\n[bold white]Style Combinations[/]")
console.print("[bold cyan]Bold Cyan[/]")
console.print("[italic green]Italic Green[/]")
console.print("[bold italic red]Bold Italic Red[/]")
console.print("[underline blue]Underline Blue[/]")
console.print("[bold underline magenta]Bold Underline Magenta[/]")

console.print("\n[bold green]Note:[/] You can also use:")
console.print("1. Any RGB color: [bold]rgb(r,g,b)[/]")
console.print("2. Any Hex color: [bold]#RRGGBB[/]")
console.print("3. Combine with backgrounds: [bold]color on background[/]")
console.print("4. Combine with styles: [bold]bold italic underline color[/]") 