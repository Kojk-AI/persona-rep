import os
import pandas as pd
from datetime import datetime

class CustomLogger:
    """
    A class to handle logging operations, centralizing the logic for consistent logging across modules.

    Examples:
        >>> logger = CustomLogger()
        >>> logger.log_to_csv('example_log.csv', event='test_event', status='success')
    """

    def __init__(self, log_directory: str = "../data/logs"):
        """
        Initialize the Logger with the given log directory.

        Args:
            log_directory (str): The directory where log files are stored.

        Examples:
            >>> logger = CustomLogger()
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_directory = os.path.join(script_dir, log_directory)
        self.ensure_directory_exists(self.log_directory)

    def get_log_filepath(self, log_file: str) -> str:
        """
        Get the full file path for a given log file.

        Args:
            log_file (str): The name of the log file.

        Returns:
            str: The full file path.

        Examples:
            >>> logger = CustomLogger()
            >>> logger.get_log_filepath('example_log.csv')
            '../data/logs/example_log.csv'
        """
        return os.path.join(self.log_directory, log_file)

    def ensure_directory_exists(self, directory: str):
        """
        Ensure that the specified directory exists, creating it if necessary.

        Args:
            directory (str): The directory to check.

        Examples:
            >>> logger = CustomLogger()
            >>> logger.ensure_directory_exists('logs')
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def log_to_csv(self, log_file: str, **kwargs):
        """
        Log key-value pairs to a CSV file, ensuring the timestamp column 'logged_time' is last.

        Args:
            log_file (str): The name of the CSV file to write to.
            **kwargs: Arbitrary keyword arguments representing the data to log.

        Examples:
            >>> logger = CustomLogger()
            >>> logger.log_to_csv(log_file='example_log.csv', event='test_event', status='success')

        The function will add a 'logged_time' column with the current timestamp
        at the end of each row and ensures that if the CSV file exists, new entries are appended, and the column order is maintained with 'logged_time' at the end.
        """
        log_directory = os.path.dirname(self.get_log_filepath(log_file))
        self.ensure_directory_exists(log_directory)
        filepath = self.get_log_filepath(log_file)
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = {**kwargs}
        df_new = pd.DataFrame([data])

        df_new['logged_time'] = now

        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath, encoding='utf-8-sig')

            if 'logged_time' in df_existing.columns:
                df_existing = df_existing[[col for col in df_existing.columns if col != 'logged_time'] + ['logged_time']]

            df_final = pd.concat([df_existing, df_new], axis=0)
            df_final = df_final.reindex(columns=pd.Index(list(df_existing.columns) + list(kwargs.keys())).unique())
        else:
            df_final = df_new

        df_final.fillna('', inplace=True)
        df_final.to_csv(filepath, index=False, encoding='utf-8-sig')