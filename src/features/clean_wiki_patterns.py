from pathlib import Path
from glob import glob
import os

pattern_to_remove = [
    '[[',
    ']]',
    '==',
    "'''",
    '{{',
    '}}',
]

def clean_file(input_fn, output_fn):
    with open(input_fn, 'r') as input_file:
        content = input_file.read()

    for pattern in pattern_to_remove:
        content = content.replace(pattern, '')

    with open(output_fn, 'w') as output_file:
        output_file.write(content)

def clean_files(source_folder, dest_folder):
    for category in os.listdir(source_folder):
        category_folder = source_folder / category
        if not category_folder.is_dir():
            continue
        os.makedirs(dest_folder / category, exist_ok = True)
        for filename in os.listdir(source_folder / category):
            input_fn  = source_folder / category / filename
            output_fn = dest_folder / category / filename
            clean_file(input_fn, output_fn)

if __name__ == '__main__':
    source_folder = Path('../../data/raw')
    dest_folder   = Path('../../data/interim')
    clean_files(source_folder, dest_folder)
