# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import re

def read_dfs(input_filepath):
    epi=pd.read_json(input_filepath+'/raw/recipes_raw/recipes_raw_nosource_epi.json', orient='index')
    fn=pd.read_json(input_filepath+'/raw/recipes_raw/recipes_raw_nosource_fn.json', orient='index')
    ar=pd.read_json(input_filepath+'/raw/recipes_raw/recipes_raw_nosource_ar.json', orient='index')
    epi['source']='epi'
    ar['source']='ar'
    fn['source']='fn'
    df=pd.concat([ar.dropna(), fn.dropna(), epi.dropna()])
    print('total dataset size:',df.shape)
    return df

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # read dataframe from csvs
    df=read_dfs(input_filepath)
    
    # clean ingredients data
    df['ingredients']=df['ingredients'].apply((lambda x: [ingred.replace('ADVERTISEMENT', '').strip() for ingred in x]))
    
    df.to_pickle(output_filepath+'/interim/merged_df_cleaned.pkl')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
