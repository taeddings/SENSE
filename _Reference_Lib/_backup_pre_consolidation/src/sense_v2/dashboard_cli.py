from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.console import Group
from rich.align import Align
import pyfiglet

# Force terminal to recognize colors properly
console = Console(force_terminal=True, color_system="truecolor")

def get_header():
    """
    Generates the Header. 
    Uses a Group to vertically stack the Logo and the Info, preventing 
    text from floating outside the box.
    """
    # 1. Generate ASCII Logo
    f = pyfiglet.Figlet(font='slant')
    logo_raw = f.renderText('SENSE')
    logo_text = Text(logo_raw, style="bold cyan")
    
    # 2. Generate Info Text (Version, Status)
    # Using Hex colors (#888888) to ensure it looks GREY, not BOLD WHITE
    info_text = Text()
    info_text.append("Autonomous Intelligence Framework v2.0\n", style="white")
    info_text.append("──────────────────────────────────────\n", style="#444444") 
    info_text.append("Version: ", style="#888888")
    info_text.append("2.0.4-alpha", style="green")
    info_text.append("   |   Modules: ", style="#888888")
    info_text.append("42 Active", style="yellow")
    info_text.append("   |   Status: ", style="#888888")
    info_text.append("ONLINE", style="bold green")

    # 3. Group them together (Centered)
    content_group = Group(
        Align.center(logo_text),
        Align.center(info_text)
    )
    
    # 4. Return Single Panel (No nesting!)
    return Panel(
        content_group,
        border_style="cyan",
        title="[bold cyan]SYSTEM DASHBOARD[/]",
        padding=(1, 1)
    )

def create_section_table(title, items, color):
    """Helper to create the inner tables for the module list."""
    table = Table(show_header=True, header_style=f"bold underline {color}", box=None, padding=(0, 1))
    table.add_column(title)
    
    for item in items:
        table.add_row(f"[{color}]{item}[/]")
    
    return table

def get_modules_panel():
    """
    Generates the Modules section.
    Uses 'Columns' which handles Termux wrapping much better than 'Grid'.
    """
    # Data
    core = [
        " 1. Kernel_Orchestrator", " 2. Memory_VectorDB", " 3. Llama_Inference_Eng",
        " 4. Context_Window_Mgr", " 5. Task_Queue_Sched", " 6. Local_Hardware_Mon",
        " 7. Python_Sandbox_Env", " 8. API_Gateway_v2"
    ]
    cog = [
        "13. NLP_Processor", "14. Sentiment_Analysis", "15. Code_Gen_Model",
        "16. RAG_Pipeline_Srch", "17. Logic_Reasoning", "18. Pattern_Recog_Net",
        "19. Image_Vision_Enc", "20. Audio_Transcriber"
    ]
    inter = [
        "25. Voice_Synthesis_TTS", "26. Discord_Gateway", "27. Telegram_Relay",
        "28. Home_Auto_Bridge", "29. Email_Notifier", "30. Web_Dashboard_React",
        "31. Mobile_Push_Svc", "32. Auth_Security_Mgr"
    ]

    # Create the 3 internal tables
    t1 = create_section_table("Core & Infrastructure", core, "cyan")
    t2 = create_section_table("Cognitive Processing", cog, "magenta")
    t3 = create_section_table("Interfaces", inter, "green")

    # Responsive Layout Logic
    # If width < 90 (Mobile Portrait), stack 1 column. 
    # Else (Landscape), use 3 columns.
    if console.width < 90:
        layout_render = Columns([t1, t2, t3], equal=True, expand=True, column_first=True)
    else:
        layout_render = Columns([t1, t2, t3], equal=True, expand=True)

    return Panel(
        layout_render,
        title="[bold cyan]Active Modules[/]",
        border_style="cyan",
        padding=(1, 1)
    )

def main():
    console.clear()
    
    # Create the main layout container
    layout = Layout()
    
    # Split layout: Header (top), Body (middle), Footer (bottom)
    layout.split_column(
        Layout(name="header", size=None), # Auto-height
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3)
    )
    
    # Populate the layout
    layout["header"].update(get_header())
    layout["body"].update(get_modules_panel())
    
    # Footer
    footer_text = Text(" sense-v2 > Type 'help' to see available commands...", style="#888888")
    layout["footer"].update(Panel(footer_text, border_style="#444444"))
    
    # Print
    console.print(layout)

if __name__ == "__main__":
    main()
