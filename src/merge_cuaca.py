import pandas as pd
import os
from glob import glob

FOLDER_CUACA = r'D:\fatiyya\KULIAH\TUGAS KULIAH\SEMESTER 6\DAMING\project-daming\data\data cuaca'

file_cuaca = glob(os.path.join(FOLDER_CUACA, '*.csv'))

all_cuaca = []

for path in file_cuaca:

    filename = os.path.basename(path)

    # skip merged lama
    if 'cuaca_merged.csv' in filename:
        continue

    station_name = filename.split(' 2022')[0]
    station_name = station_name.replace('_', ' ').lower()

    print(f'\nProcessing: {station_name}')

    try:

        # READ CSV NORMAL
        df_temp = pd.read_csv(path)

        # tambah station
        df_temp['station'] = station_name

        # datetime
        df_temp['datetime'] = pd.to_datetime(df_temp['datetime'])

        print(df_temp.shape)

        all_cuaca.append(df_temp)

    except Exception as e:

        print(f'ERROR: {e}')

# gabungkan semua
df_cuaca_merged = pd.concat(all_cuaca, ignore_index=True)

# pindahkan station ke depan
cols = ['station'] + [c for c in df_cuaca_merged.columns if c != 'station']

df_cuaca_merged = df_cuaca_merged[cols]

print('\nHASIL AKHIR')
print(df_cuaca_merged.shape)

print('\nUNIQUE STATION:')
print(df_cuaca_merged['station'].unique())

# save
SAVE_PATH = r'D:\fatiyya\KULIAH\TUGAS KULIAH\SEMESTER 6\DAMING\project-daming\data\data cuaca\cuaca_merged.csv'

mask_us1 = df_cuaca_merged['station'] == 'us embassy 1'

kolom_fahrenheit = [
    'tempmax',
    'tempmin',
    'temp',
    'feelslikemax',
    'feelslikemin',
    'feelslike',
    'dew'
]

for col in kolom_fahrenheit:
    df_cuaca_merged.loc[mask_us1, col] = (
        (df_cuaca_merged.loc[mask_us1, col] - 32) * 5/9
    )

df_cuaca_merged.to_csv(SAVE_PATH, index=False)

print('\nBERHASIL DISIMPAN')

