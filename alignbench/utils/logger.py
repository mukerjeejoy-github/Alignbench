# alignbench/utils/logger.py

from termcolor import colored

def info(message):
    print(colored(f"[INFO] {message}", "cyan"))

def success(message):
    print(colored(f"[SUCCESS] {message}", "green"))

def warning(message):
    print(colored(f"[WARNING] {message}", "yellow"))

def error(message):
    print(colored(f"[ERROR] {message}", "red"))
