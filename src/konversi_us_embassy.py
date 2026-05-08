mask_us1 = df_cuaca['station'] == 'us embassy 1'

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
    df_cuaca.loc[mask_us1, col] = (
        (df_cuaca.loc[mask_us1, col] - 32) * 5/9
    )

df_cuaca.to_csv(SAVE_PATH, index=False)