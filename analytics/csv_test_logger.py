import csv
import time
from typing import List, TypedDict, Union

from analytics.csv_logger import CSVLogger


class EpochLogEntry(TypedDict):
    epoch: int
    psnr: float
    ssim: float


class CSVTestLogger(CSVLogger):
    """
    Logger for writing and reading training logs to CSV files.
    Supports batch-level and epoch-level logging.
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self.headers = ["epoch", "psnr", "ssim"]

        self._ensure_csv(self.log_path, self.headers)

    def log_epoch(self, epoch: int, psnr: float, ssim: float) -> None:
        """
        Log an epoch entry. Overwrites any previous entry for the same epoch.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._overwrite_or_append(
            filepath=self.log_path,
            headers=self.headers,
            new_row=[timestamp, epoch, psnr, ssim],
            match_keys=[1],  # epoch
        )

    def read_epoch_logs(self) -> List[EpochLogEntry]:
        """Read epoch log entries from file."""
        logs: List[EpochLogEntry] = []
        with open(self.log_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(EpochLogEntry(epoch=int(row["epoch"]), psnr=float(row["psnr"]), ssim=float(row["ssim"])))
        return logs
