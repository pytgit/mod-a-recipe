import json
import re
import string
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
import code

import utils

from sklearn.model_selection import train_test_split

import spacy
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

# to track time
from time import time

def parseNumbers(s):
    """
    Parses a string that represents a number into a decimal data type so that
    we can match the quantity field in the db with the quantity that appears
    in the display name. Rounds the result to 2 places.
    """
    ss = utils.unclump(s)

    # check for int + fractions
    m1 = re.match(r'(\d+)\s+(\d)/(\d)', ss)
    if m1 is not None:
        num = int(m1.group(1)) + (float(m1.group(2)) / float(m1.group(3)))
        #return decimal.Decimal(str(round(num,2)))
        return m1.group(1)+' '+m1.group(2)+'/'+m1.group(3)
    
    # check for fractions
    m2 = re.match(r'(\d)/(\d)', ss)
    if m2 is not None:
        num = float(m2.group(1)) / float(m2.group(2))
        #return decimal.Decimal(str(round(num,2)))
        return m2.group(1)+'/'+m2.group(2)
    
    # check for integers
    m3 = re.match(r'(\d+)', ss)
    if m3 is not None:
        #return decimal.Decimal(round(float(ss), 2))
        return m3.group(1)

    return None

def convert_to_spacy(df):
    try:
        # input, name, qty, range_end, unit, comment
        spacy_training=[]
        labels=['name', 'qty', 'unit', 'comment']
        
        for index, row in df.iterrows():
            entities = []
            text=str(row['input'])
            for label in labels:
                entity=str(row[label])
                start=text.find(entity)
                if (label=='qty') & (start==-1):
                    entity=parseNumbers(text)
                    if entity:
                        start=text.find(entity)
                    #print('parsed:', entity)            
                if (start!=-1):
                    end=start+len(entity)
                    entities.append((start, end,label))
                    #print(text, entity, start, end)    
            spacy_training.append((text, {"entities" : entities}))

        return spacy_training
    except Exception as e:
        logging.exception("Unable to process \n" + "error = " + str(e))
        return None

# Custom tokenizer for ingredients to allow for hyphenated words in ingredients (e.g. extra-virgin olive oil)
def ingred_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\*\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

def find_ingred_name(text):
    ingred_to_test=nlp(text)
    name=""
    
    for ent in ingred_to_test.ents:
        if ent.label_=='name':
            name=ent.text.lower().strip()
    
    if (len(name)>0):
        tokens=nlp2(name)
        for token in tokens: 
            #print('Tokens:',token.text,token.tag_)
            #print('Before pos:', text)
            if token.text.lower() in SYMBOLS:
                name=name.replace(token.text, "")
            if (token.tag_=='NNS'):
                name=name.replace(token.text, token.lemma_.lower())
                #print('NNS parsed:',text)

        return name
    
    return None

# Extract cooking time by finding all acounts of time and units in cooking instructions
def extract_time(instructions):
    # convert time strings to time to compute total minutes in instructions
    conv_sec=1/60
    conv_dict={'day': 1440, 'days': 1440, 
               'hour': 60, 'hr': 60, 'hrs':60, 'hours':60, 
               'mins':1, 'min': 1, 'minute':1, 'minutes': 1, 
               'second': conv_sec, 'sec': conv_sec, 'seconds': conv_sec, 'secs': conv_sec}

    p = re.compile('([0-9]{1,2}\s\w+)', re.IGNORECASE)
    matches=p.findall(instructions)
    
    total_in_min=0
    for m in matches:
        [time, unit]=m.split()
        # check for valid time unit
        if unit in conv_dict:
            total_in_min+=float(time)*conv_dict[unit]
    
    removed_text=re.sub(p, "", instructions)
    return removed_text, total_in_min

# MAIN FUNCTION for loading cleaned dataframe and extracting features
def main():
    logger = logging.getLogger(__name__)

    # read dataframe
    logger.info('Reading data...')
    df=pd.read_pickle('../../data/interim/merged_df_cleaned.pkl')
    
    # create cleaned instructions and total cooking time
    logger.info('Extract cooking time...')
    df['instructions_c'], df['total_time'] = zip(*df['instructions'].apply(extract_time))

    # merge title and instructions into one feature
    df['title_instruct']=df['title'].astype(str) + '\n'+ df['instructions_c']
    
    # extract ingredients
    logger.info('Start extracting ingredients...')
    tqdm.pandas()
    
    df['ingred_s']=df['ingredients'].progress_apply(lambda x: [find_ingred_name(ingredients) for ingredients in x])
    # get rid of ingredients that are "None"
    df['ingred_s']=df['ingred_s'].apply(lambda x: [item for item in x if item is not None])
    
    # pickle dataframe
    df.to_pickle('../../data/processed/df_for_model_spacy.pkl')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    
    # use previously trained ingredient entities NLP NER model
    nlp = spacy.load('../../models/nlp_model') 
    # initialize NLP tokenizer and parser for further ingredients cleaning
    nlp2 = spacy.load('en')
    nlp2.tokenizer = ingred_tokenizer(nlp2)
    
    SYMBOLS=" ".join(string.punctuation).split(" ") + ["®","*", "/","-", "-", "...", "”", "'"]

    
    main()