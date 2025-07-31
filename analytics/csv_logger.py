import csv
import os
from typing import List, Union


class CSVLogger:
    """
    Base Logger for writing and reading logs to CSV files.
    Contains help methods to be used by children loggers.
    """

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
