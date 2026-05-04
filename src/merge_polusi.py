import pandas as pd
import os
import re

BASE_DIR = "data/data polusi"

def extract_date_from_filename(filename):
    # ambil format YYYY-MM-DD dari nama file
    match = re.search(r"\d{4}-\d{2}-\d{2}", filename)
    if match:
        return match.group(0)
    return None

def merge_all_stations(base_dir):
    all_data = []

    for station_folder in os.listdir(base_dir):
        station_path = os.path.join(base_dir, station_folder)

        if os.path.isdir(station_path):
            print(f"Processing: {station_folder}")

            for file in os.listdir(station_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(station_path, file)

                    try:
                        df = pd.read_csv(file_path)

                        # 🔥 ambil tanggal dari nama file
                        tanggal = extract_date_from_filename(file)

                        # tambahkan kolom
                        df["tanggal"] = tanggal
                        df["station"] = station_folder

                        all_data.append(df)

                    except Exception as e:
                        print(f"Error: {file} -> {e}")

    df_merged = pd.concat(all_data, ignore_index=True)

    return df_merged


if __name__ == "__main__":
    df_polusi = merge_all_stations(BASE_DIR)

    output_path = "data/merged data polusi/polusi_merged.csv"
    df_polusi.to_csv(output_path, index=False)

