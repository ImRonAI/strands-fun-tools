"""Example agent using Strands Fun Tools.

This agent has access to creative and interactive tools including:
- Bluetooth & WiFi monitoring
- Chess engine
- Clipboard management
- Cursor control
- Computer vision (YOLO, face recognition)
- Screen reading
- Audio transcription
- And more!
"""

from strands import Agent
from strands_fun_tools import (
    bluetooth,
    chess,
    clipboard,
    cursor,
    screen_reader,
    yolo_vision,
    human_typer,
    spinner_generator,
    template,
    utility,
)

# Create agent with fun tools
agent = Agent(
    tools=[
        bluetooth,
        chess,
        clipboard,
        cursor,
        screen_reader,
        yolo_vision,
        human_typer,
        spinner_generator,
        template,
        utility,
    ],
    system_prompt="""You are a creative AI agent with unique interactive capabilities.

You can:
- Monitor Bluetooth devices and detect proximity
- Play chess using the Stockfish engine
- Monitor and control the system clipboard
- Control the mouse cursor and automate tasks
- Read the screen using OCR
- Detect objects using YOLO vision
- Type with human-like characteristics
- Display custom loading spinners
- Render Jinja2 templates
- Perform cryptography and encoding operations

Be creative and explore these capabilities!""",
)

if __name__ == "__main__":
    # Example usage
    response = agent(
        """Show me what you can do! Start by:
1. Scanning for Bluetooth devices nearby
2. Getting my clipboard content
3. Detecting what's on my screen"""
    )
    print(response)
