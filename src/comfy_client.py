import configparser
import subprocess


def run_comfy_client():
    config = configparser.ConfigParser()
    config.read("config.properties", encoding='utf8')

    if config["BOT"]["USE_EMBEDDED_COMFY"].lower() == "true":
        import os
        print("Starting embedded comfy")
        comfy_path = "embedded_comfy"
        if os.name == 'nt':
            subprocess.Popen(["./venv/Scripts/python.exe", "main.py", "--port", config["EMBEDDED"]["SERVER_PORT"], "--listen", "--preview-method", "auto"], cwd=comfy_path)
        else:
            subprocess.Popen(["../venv/bin/python", "main.py", "--port", config["EMBEDDED"]["SERVER_PORT"], "--listen", "--preview-method", "auto"], cwd=comfy_path)
    else:
        print(f"Using external comfy server. Make sure it's running. Address: {config['LOCAL']['SERVER_ADDRESS']}")
