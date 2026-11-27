import os
import glob
import time
import platform
import subprocess
from datetime import datetime
import psutil
import pyautogui

# 1. Calculator Tool (Opens Windows Calculator & types calculation)
def open_calculator(expression: str) -> str:
    subprocess.Popen("calc.exe")
    time.sleep(1.5)
    clean_expr = expression.replace("×", "*").replace("÷", "/").replace(" ", "")
    pyautogui.write(clean_expr + "=", interval=0.1)
    try:
        result = eval(clean_expr)
    except Exception:
        result = "Displayed in Windows Calculator"
    return f"[Calculator Opened] Typed '{clean_expr}=' -> Result: {result}"

# 2. Date and Time Tool
def get_datetime() -> str:
    return f"Current Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# 3. Notepad Tool (Opens Windows Notepad & writes text)
def open_notepad(text: str) -> str:
    subprocess.Popen("notepad.exe")
    time.sleep(1.5)
    pyautogui.write(text, interval=0.05)
    return f"[Notepad Opened] Typed: '{text}'"

# 4. File Search Tool
def search_files(directory: str, pattern: str = "*.pdf") -> str:
    if not os.path.exists(directory):
        directory = "."
    matches = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
    return f"Found {len(matches)} file(s) matching '{pattern}' in '{directory}': {matches[:5]}"

# 5. File Creation Tool
def create_text_file(filepath: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Created file '{filepath}' with content: '{content}'"

# 6. System Information Tool
def get_system_info() -> str:
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return (
        f"OS: {platform.system()} {platform.release()} | "
        f"Processor: {platform.processor()} | "
        f"Total RAM: {ram_gb} GB"
    )

# Agent Router & Execution Loop
def run_tool_agent(query: str):
    q = query.lower()
    print(f"\nUser Query: {query}")
    if "calculate" in q or "×" in q or "*" in q:
        expr = query.lower().replace("calculate", "").strip()
        print("[TOOL SELECTED] open_calculator")
        print("[TOOL RESULT]", open_calculator(expr))
    elif "date" in q or "time" in q:
        print("[TOOL SELECTED] get_datetime")
        print("[TOOL RESULT]", get_datetime())
    elif "notepad" in q:
        text = query.split("write", 1)[-1].strip() if "write" in q else "Hello Students"
        print("[TOOL SELECTED] open_notepad")
        print("[TOOL RESULT]", open_notepad(text))
    elif "find" in q and "pdf" in q:
        print("[TOOL SELECTED] search_files")
        print("[TOOL RESULT]", search_files("..", "*.pdf"))
    elif "create a file" in q:
        print("[TOOL SELECTED] create_text_file")
        print("[TOOL RESULT]", create_text_file("test.txt", "Hello World"))
    elif "specifications" in q or "system" in q:
        print("[TOOL SELECTED] get_system_info")
        print("[TOOL RESULT]", get_system_info())

if __name__ == "__main__":
    # Execute all 6 test queries from Lab Sheet 6 Exercise
    test_queries = [
        "Calculate 25 × 4",
        "What is the date today?",
        "Open Notepad and write Hello Students",
        "Find all PDF files in workspace",
        "Create a file called test.txt with Hello World",
        "Show my computer specifications"
    ]
    for query in test_queries:
        run_tool_agent(query)
        time.sleep(1)
