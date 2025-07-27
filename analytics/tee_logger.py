import sys


class TeeLogger:
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        # Needed for Python's internal buffering compatibility
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()
