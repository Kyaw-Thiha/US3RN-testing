# Ensure you are running this file from root.
# python -m train_analyze.import_from_log
#
# Otherwise, change the file paths to be relative

import re

from analytics.csv_train_logger import CSVTrainLogger

batch_log_file = "analytics/batch_logs.csv"
epoch_log_file = "analytics/epoch_logs.csv"
log_file = "logs/train_logs/train_10.log"


def import_log_file(log_path: str, logger: CSVTrainLogger) -> None:
    """
    Reads a training log file (e.g., train_10.log) and parses batch and epoch loss entries,
    saving them to the batch and epoch CSV logs.
    """
    with open(log_path, "r") as f:
        for line in f:
            # Match batch lines like: ===> Epoch[1](100/2500): Loss: 1.5179
            batch_match = re.match(r"===> Epoch\[(\d+)\]\((\d+)/\d+\): Loss: ([\d.]+)", line)
            if batch_match:
                print(f"Logging {batch_match} onto csv")
                epoch = int(batch_match.group(1))
                batch = int(batch_match.group(2))
                loss = float(batch_match.group(3))
                logger.log_batch(epoch, batch, loss)
                continue

            # Match epoch average loss line like: ===> Epoch 1 Complete: Avg. Loss: 481310018.2940
            epoch_match = re.match(r"===> Epoch (\d+) Complete: Avg. Loss: ([\d.]+)", line)
            if epoch_match:
                print(f"Logging {epoch_match} onto csv")
                epoch = int(epoch_match.group(1))
                loss = float(epoch_match.group(2))
                logger.log_epoch(epoch, loss)
            print("-------------------------------------")


if __name__ == "__main__":
    print(f"Loading the Logger with {batch_log_file} and {epoch_log_file}")
    logger = CSVTrainLogger(batch_log_file, epoch_log_file)

    print("Importing the logs from the .log file")
    import_log_file(log_file, logger)

    print("Finished logging")
