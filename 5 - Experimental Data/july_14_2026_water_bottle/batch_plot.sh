#!/bin/bash

# Find all files ending in .csv in the current directory and subdirectories
# excluding any folder that might have "figures" in its name to avoid infinite loops
find . -type f -name "*.csv" ! -path "*/_figures/*" | while read -r csv_file; do
    echo "Processing: $csv_file"
    python3 plot_flight_data.py "$csv_file"
done

echo "All CSV files have been processed."