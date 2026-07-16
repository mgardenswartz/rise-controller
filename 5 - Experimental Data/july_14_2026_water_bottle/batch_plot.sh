#!/bin/bash

# Find all .csv files excluding "venv" and "_figures" directories
find . -type f -name "*.csv" ! -path "*/venv/*" ! -path "*/_figures/*" | while read -r csv_file; do
    echo "Processing: $csv_file"
    python3 plot_flight_data.py "$csv_file"
done

echo "All CSV files have been processed."