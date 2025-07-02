import os
import re
import argparse
from datetime import datetime
import pandas as pd

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Process NMR audit trail and compute relative end times.")
parser.add_argument("base_dir", help="Path to base directory containing experiment folders")
parser.add_argument("-off", "--offset", type=float, default=0, help="Manual offset (seconds) to add to each end time")
args = parser.parse_args()

base_dir = args.base_dir
manual_offset = args.offset

# --- Regex patterns ---
start_pattern = re.compile(r"started at ([\d\-:.\s\+]+)")
end_pattern = re.compile(r"completed at ([\d\-:.\s\+]+)")

results = []

# --- Scan folders ---
for folder in os.listdir(base_dir):
    try:
        if int(folder) >= 10:
            folder_path = os.path.join(base_dir, folder)
            file_path = os.path.join(folder_path, "audita.txt")

            if os.path.isdir(folder_path) and os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    contents = file.read()
                    start_match = start_pattern.search(contents)
                    end_match = end_pattern.search(contents)
                    if end_match:
                        end_time_str = end_match.group(1).strip()
                        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S.%f %z")

                        if start_match:
                            start_time_str = start_match.group(1).strip()
                            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f %z")
                        else:
                            start_time = None

                        results.append((folder, start_time, end_time))
    except ValueError:
        continue

# --- Sort results numerically ---
results.sort(key=lambda x: int(x[0]))

# --- Prepare DataFrame ---
if results:
    base_time = results[0][1] if results[0][1] else results[0][2]  # Prefer start time of first folder

    data = []
    for folder, start_time, end_time in results:
        delta_seconds = (end_time - base_time).total_seconds() + manual_offset
        data.append({
            'Folder': folder,
            'Start Time': start_time.strftime("%Y-%m-%d %H:%M:%S.%f %z") if start_time else "",
            'End Time': end_time.strftime("%Y-%m-%d %H:%M:%S.%f %z"),
            'Relative End Time (s)': round(delta_seconds, 3)
        })

    df = pd.DataFrame(data)

    # --- Use folder name for output filename (no extra suffix) ---
    parent_dir = os.path.dirname(base_dir)  # Get the parent directory of the base directory
    directory_name = os.path.basename(base_dir)  # Get the base folder name (not full path)
    output_filename = f"{directory_name}_relative_end_times.xlsx"  # Append the "_relative_end_times" suffix
    output_path = os.path.join(parent_dir, output_filename)  # Save it in the parent directory

    # --- Save to Excel ---
    df.to_excel(output_path, index=False)
    print(f"\n✅ Saved to Excel: {output_path}")
else:
    print("No valid end times found.")
