# 🎨 Strands Fun Tools

[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

Creative and utility tools for Strands AI agents - Bluetooth, vision, cursor control, audio, and more!

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 🔌 Use as an MCP server

Use strands-fun-tools from **Claude Code, Claude Desktop, Cursor, Kiro, or any MCP client** — every available tool (`cursor`, `clipboard`, `chess`, `bluetooth`, `screen_reader`, `yolo_vision`, `take_photo`, `listen`, `dialog`, and more) becomes an MCP tool.

```bash
claude mcp add fun -- uvx strands-fun-tools
```

Claude Desktop config:

```json
{
  "mcpServers": {
    "fun": {
      "command": "uvx",
      "args": ["strands-fun-tools"]
    }
  }
}
```

Options:

```bash
strands-fun-tools --tools cursor,clipboard,take_photo   # expose a subset
strands-fun-tools --skip bluetooth,chess                # drop groups
strands-fun-tools --http --port 8000                    # HTTP mode, multi-client
```

> Tool groups import lazily — groups whose optional deps aren't installed are skipped automatically. Install extras for more tools: `uvx --with 'strands-fun-tools[all]' strands-fun-tools`

---

## 📦 Installation

```bash
# Base installation (human_typer only)
pip install strands-fun-tools

# With specific features
pip install "strands-fun-tools[cursor,clipboard,vision]"

# Everything
pip install "strands-fun-tools[all]"
```

## 🛠️ Available Tools

### 🎮 Interaction
- **human_typer** - Human-like typing with emotions and typos
- **cursor** - Mouse & keyboard automation (pyautogui)
- **clipboard** - Clipboard monitoring & history
- **dialog** - Interactive terminal prompts

### 👁️ Vision
- **screen_reader** - OCR-based screen monitoring
- **yolo_vision** - Real-time object detection (YOLOv8)
- **face_recognition** - Face detection via AWS Rekognition
- **take_photo** - Camera capture & burst mode

### 🎤 Audio
- **listen** - Background audio transcription (Whisper)

### 📡 Connectivity
- **bluetooth** - BLE device monitoring & GATT operations

### ♟️ Games
- **chess** - Stockfish chess engine integration

### 🎨 Display
- **spinner_generator** - Custom loading animations
- **template** - Jinja2 template rendering
- **asciimatics_ui** - Terminal UI framework

### 🔧 Utilities
- **utility** - Crypto, encoding, hashing, JSON/YAML
- **dynamic_package** - Execute any Python package function
- **npm** - Run npm packages from Python

## 🚀 Quick Start

```python
from strands import Agent
from strands_fun_tools import human_typer, cursor, clipboard

agent = Agent(
    tools=[human_typer, cursor, clipboard],
    system_prompt="You can type like a human and control the cursor!"
)

agent("Type 'Hello World!' with excited emotion and then copy it to clipboard")
```

## 📋 Tool Reference

| Tool | Install Extra | Key Actions |
|------|---------------|-------------|
| **human_typer** | *(base)* | Type with emotions: calm, excited, thoughtful, rushed, nervous |
| **cursor** | `[cursor]` | move, click, drag, type_text, hotkey |
| **clipboard** | `[clipboard]` | start, read, write, get_history |
| **screen_reader** | `[vision]` | start, capture_once, find_element |
| **yolo_vision** | `[vision]` | start, detect_once, query_objects |
| **face_recognition** | `[face]` | detect_faces, compare_faces |
| **take_photo** | `[vision]` | capture, burst, list_cameras |
| **listen** | `[audio]` | start, stop, get_transcripts |
| **bluetooth** | `[bluetooth]` | start, scan_once, list_devices, read_characteristic |
| **chess** | `[chess]` | new_game, get_best_move, make_move, analyze |
| **spinner_generator** | `[display]` | Display 50+ spinner types |
| **template** | `[template]` | create, render Jinja2 templates |
| **utility** | `[utility]` | encode, decode, hash, encrypt |
| **dynamic_package** | *(base)* | execute any Python function |
| **npm** | *(base)* | execute npm packages |
| **dialog** | `[dialog]` | Interactive terminal prompts |
| **asciimatics_ui** | `[ui]` | Terminal UI framework |

## 🎯 Examples

### Background Monitoring
```python
from strands import Agent
from strands_fun_tools import bluetooth, clipboard, yolo_vision

agent = Agent(tools=[bluetooth, clipboard, yolo_vision])

agent("""
Start monitoring:
1. Bluetooth devices nearby
2. Clipboard content changes  
3. Objects visible on camera
""")
```

### Autonomous Screen Control
```python
from strands import Agent
from strands_fun_tools import screen_reader, cursor

agent = Agent(tools=[screen_reader, cursor])

agent("""
1. Find the 'Submit' button on screen
2. Click it
""")
```

### Human-Like Typing
```python
agent.tool.human_typer(
    text="Hello World!",
    emotion="excited",      # calm, excited, thoughtful, rushed, nervous
    typo_rate=2,           # 0-10 percentage
    thinking_pauses=True   # Pause at punctuation
)
```

## 📚 Documentation

Full documentation at [github.com/cagataycali/strands-fun-tools](https://github.com/cagataycali/strands-fun-tools)

## 🤝 Contributing

Issues and PRs welcome!

```bash
git clone https://github.com/cagataycali/strands-fun-tools.git
cd strands-fun-tools
pip install -e ".[all,dev]"
```

## 📄 License

Apache-2.0 - see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ for the Strands community
</div>
