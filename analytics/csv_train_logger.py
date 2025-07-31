import csv
import time
from typing import List, TypedDict, Union

from analytics.csv_logger import CSVLogger


class BatchLogEntry(TypedDict):
    timestamp: str
    epoch: int
    batch: int
    loss: float


class EpochLogEntry(TypedDict):
    timestamp: str
    epoch: int
    loss: float


class CSVTrainLogger(CSVLogger):
    """
    Logger for writing and reading training logs to CSV files.
    Supports batch-level and epoch-level logging.
    """

    def __init__(self, batch_log_path: str, epoch_log_path: str) -> None:
        self.batch_log_path = batch_log_path
        self.epoch_log_path = epoch_log_path
        self.batch_headers = ["timestamp", "epoch", "batch", "loss"]
        self.epoch_headers = ["timestamp", "epoch", "loss"]

        self._ensure_csv(self.batch_log_path, self.batch_headers)
        self._ensure_csv(self.epoch_log_path, self.epoch_headers)

    def log_batch(self, epoch: int, batch: int, loss: float) -> None:
        """
        Log a batch entry. Overwrites any previous entry for the same (epoch, batch).
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._overwrite_or_append(
            filepath=self.batch_log_path,
            headers=self.batch_headers,
            new_row=[timestamp, epoch, batch, loss],
            match_keys=[1, 2],  # epoch, batch
        )

    def log_epoch(self, epoch: int, loss: float) -> None:
        """
        Log an epoch entry. Overwrites any previous entry for the same epoch.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self._overwrite_or_append(
            filepath=self.epoch_log_path,
            headers=self.epoch_headers,
            new_row=[timestamp, epoch, loss],
            match_keys=[1],  # epoch
        )

    def read_batch_logs(self) -> List[BatchLogEntry]:
        """Read batch log entries from file."""
        logs: List[BatchLogEntry] = []
        with open(self.batch_log_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(
                    BatchLogEntry(
                        timestamp=row["timestamp"], epoch=int(row["epoch"]), batch=int(row["batch"]), loss=float(row["loss"])
                    )
                )
        return logs

    def read_epoch_logs(self) -> List[EpochLogEntry]:
        """Read epoch log entries from file."""
        logs: List[EpochLogEntry] = []
        with open(self.epoch_log_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logs.append(EpochLogEntry(timestamp=row["timestamp"], epoch=int(row["epoch"]), loss=float(row["loss"])))
        return logs
