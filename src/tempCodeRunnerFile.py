PATH = r'D:\fatiyya\KULIAH\TUGAS KULIAH\SEMESTER 6\DAMING\project-daming\data\data cuaca\bundaran_hi 2022-01-01 to 2024-12-31.csv'

with open(PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()

header = lines[0]

fixed_lines = [header]

for line in lines[1:]:

    line = line.strip()

    parts = line.split(',')

    # gabungkan lat lon jadi 1 kolom
    first_col = f'"{parts[0].strip()}, {parts[1].strip()}"'

    new_line = ','.join([first_col] + parts[2:])

    fixed_lines.append(new_line + '\n')

with open(PATH, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print('Bundaran HI fixed!')