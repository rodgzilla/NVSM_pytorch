import os
import logging
# -*- coding: utf-8 -*-
from pathlib import Path
import click


import pandas as pd

@click.command()
@click.argument('dataset_path', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(dataset_path, output_filepath):
    """ Creates a CSV file that contains a description of the
    dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info('creating dataset description')
    dataset_path = Path(dataset_path)
    entry_dicts = []
    for category in os.listdir(dataset_path):
        category_path = dataset_path / category
        if not category_path.is_dir():
            continue
        for fn in os.listdir(category_path):
            entry_dict = {
                'filename': fn,
                'category': category,
                'size'    : os.path.getsize(dataset_path / category / fn)
            }
            entry_dicts.append(entry_dict)
    df = pd.DataFrame.from_dict(entry_dicts)
    df.to_csv(output_filepath, index = False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
