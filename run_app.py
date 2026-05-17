from pathlib import Path

import subprocess
import sys


def run_command(command):
    subprocess.run(command, check=True)


def main():
    python = sys.executable

    required_files = [
        Path("app.py"),
        Path("requirements.txt"),
        Path("models") / "price_prediction_model.pkl"
    ]

    for file_name in required_files:
        if not Path(file_name).exists():
            print(f"Ошибка: файл {file_name} не найден.")
            input("Нажмите Enter для выхода...")
            sys.exit(1)

    print("Установка зависимостей...")
    run_command([python, "-m", "pip", "install", "-r", "requirements.txt"])

    print("Запуск Streamlit-приложения...")
    run_command([python, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()