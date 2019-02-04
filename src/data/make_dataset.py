# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import requests
from xml.etree import ElementTree
import time
import re
from collections import defaultdict
import os

def extract_see_also(page_text):
    try:
        lines         = page_text.splitlines()
        see_index     = lines.index('==See also==')
        ref_index     = lines.index('==References==')
        raw_titles    = lines[see_index+1:ref_index-1]
        regex         = '.*\[\[(.*)\]\]'
        parsed_titles = []
        for title in raw_titles:
            re_result = re.search(regex, title)
            if re_result:
                parsed_titles.append(re_result.group(1))

        return parsed_titles
    except:
        return []

def generate_page_name_from_title(title):
    return '_'.join(title.split())

def get_wikipedia_page(page_title, delay = 3):
    try:
        api_url           = f'https://en.wikipedia.org/wiki/Special:Export/{page_title}'
        req               = requests.get(api_url)
        time.sleep(5)
        page_text         = req.text
        xml_root          = ElementTree.fromstring(page_text)
        page_content      = xml_root\
                .find('{http://www.mediawiki.org/xml/export-0.10/}page')\
                .find('{http://www.mediawiki.org/xml/export-0.10/}revision')\
                .find('{http://www.mediawiki.org/xml/export-0.10/}text')
        page_content_text = page_content.text
        see_also_titles   = extract_see_also(page_content_text)
        see_also_links    = [generate_page_name_from_title(title) for title in see_also_titles]
        page_dict         = {
            'title'   : page_title,
            'content' : page_content_text,
            'see_also': see_also_links
        }

        return page_dict
    except:
        print(f'Problem downloading {page_title}')
        return None

def mine_graph(entry_points, n = 10):
    queues     = [[point] for point in entry_points]
    downloaded = set()
    i          = 0
    documents  = defaultdict(list)

    while len(downloaded) < n:
        if not any(queue for queue in queues):
            print('all queues are empty, exiting.')
            break
        print(100 * len(downloaded) / n, '%')
        queue       = queues[i % len(queues)]
        i          += 1
        if not queue:
            continue
        page_title, category  = queue.pop(0)
        if page_title in downloaded:
            print(f'{page_title} already downloaded')
            continue
        downloaded.add(page_title)
        page_dict = get_wikipedia_page(page_title)

        if page_dict is None:
            continue

        documents[category].append(page_dict)
        new_queue_elems = [(title, category) for title in page_dict['see_also']]
        queue.extend(new_queue_elems)

    return documents

def save_documents(documents, data_folder = Path('../data/')):
    for category in documents:
        os.makedirs(data_folder / category, exist_ok = True)
        for page in documents[category]:
            title   = page['title'].replace('/', '_')
            content = page['content']
            with open(data_folder / category / title, 'w') as page_file:
                page_file.write(content)

@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('mining raw dataset from wikipedia')
    entry_points = [
        ('French_Revolution'                       , 'history'),
        ('Aleppo_offensive_(October–December_2013)', 'history'),
        ('World_War_II'                            , 'history'),
        ('Algebraic_graph_theory'                  , 'math'),
        ('Machine_learning'                        , 'math'),
        ('Game_theory'                             , 'math'),
        ('Astronomy'                               , 'space'),
        ('Universe'                                , 'space'),
        ('Pluto'                                   , 'space'),
        ('Linguistics'                             , 'language'),
        ('Translation'                             , 'language'),
        ('Toki_Pona'                               , 'language'),
        ('Napster'                                 , 'tech'),
        ('Freenet'                                 , 'tech'),
        ('Neuralink'                               , 'tech'),
        ('For_the_World'                           , 'music'),
        ('Pixies'                                  , 'music'),
        ('Jazz'                                    , 'music'),
    ]
    documents = mine_graph(entry_points, 2000)
    save_documents(documents, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
