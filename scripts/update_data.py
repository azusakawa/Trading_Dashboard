"""
This script is designed to perform scheduled data updates for both futures and forex data.
It leverages the `data_updater` module to fetch the latest financial data
and records the timestamp of the last successful update.

This script is intended to be run periodically (e.g., via a cron job or task scheduler)
to ensure the application always has fresh data for predictions.
"""

import sys
import os

# Add the parent directory (project root) to the sys.path.
# This allows importing modules from other top-level directories like 'data' and 'utils'.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the data update functions from the data_updater module.
from data.data_updater import update_futures_data, update_forex_data

# Removed logging imports and configuration as per previous changes.

if __name__ == "__main__":
    print("Starting scheduled data update.")
    try:
        # Call the function to update futures data.
        update_futures_data()
        # Call the function to update forex data.
        update_forex_data()
        
        # Record the timestamp of the successful update.
        from datetime import datetime
        # Construct the path to the last_updated.txt file within the data directory.
        timestamp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'last_updated.txt')
        with open(timestamp_path, 'w') as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print("Scheduled data update finished successfully.")
    except Exception as e:
        # Catch and print any errors that occur during the update process.
        print(f"Error during scheduled data update: {e}")
