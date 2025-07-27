import csv
import os
import time
from typing import List, TypedDict, Union


class BatchLogEntry(TypedDict):
    timestamp: str
    epoch: int
    batch: int
    loss: float


class EpochLogEntry(TypedDict):
    timestamp: str
    epoch: int
    loss: float


class CSVLogger:
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

    def _ensure_csv(self, filepath: str, headers: List[str]) -> None:
        """Ensure the CSV file exists and has the correct header."""
        file_exists = os.path.isfile(filepath)
        needs_header = True

        if file_exists:
            with open(filepath, "r") as f:
                first_line = f.readline()
                needs_header = not first_line.strip().startswith(",".join(headers))

        if not file_exists or needs_header:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _overwrite_or_append(
        self, filepath: str, headers: List[str], new_row: List[Union[str, int, float]], match_keys: List[int]
    ) -> None:
        """
        Overwrite or append a row based on match_keys (column indices).
        """
        updated = False
        rows: List[List[str]] = []

        if os.path.isfile(filepath):
            with open(filepath, "r", newline="") as f:
                reader = csv.reader(f)
                file_headers = next(reader)
                for row in reader:
                    if all(row[i] == str(new_row[i]) for i in match_keys):
                        rows.append([str(v) for v in new_row])
                        updated = True
                    else:
                        rows.append(row)

        if not updated:
            rows.append([str(v) for v in new_row])

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
