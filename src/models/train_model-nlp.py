import json
import random
import logging
import spacy
import pickle

import re
import decimal
import optparse
import pandas as pd

import utils

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score

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
        labels=['name', 'qty', 'unit']
        
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
    
def train_spacy(nlp, TRAIN_DATA):
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
       
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)

#test the model and evaluate it
def test_results(nlp, spacy_test) :

    # initialize dictionary
    d={'name':[0,0,0,0,0], 'qty':[0,0,0,0,0], 'unit':[0,0,0,0,0]}

    # write results to file
    filepath='../../reports/nlp_model_'+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))+'.txt'
    with open(filepath, 'w') as f:
        for text,annot in spacy_test:
            ingred_to_test=nlp(text)

            for ent in ingred_to_test.ents:
                doc_gold_text= nlp.make_doc(text)
                gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
                y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
                y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in ingred_to_test]  

                #print("For Entity "+ent.label_+"\n")   
                #print(classification_report(y_true, y_pred)+"\n")
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)

                d[ent.label_][0]+=p
                d[ent.label_][1]+=r
                d[ent.label_][2]+=f
                d[ent.label_][3]+=a
                d[ent.label_][4]+=1

        for i in d:
            f.write("\n For Entity "+i+"\n")
            f.write("Accuracy : "+str((d[i][3]/d[i][4])*100)+"%")
            f.write("Precision : "+str(d[i][0]/d[i][4]))
            f.write("Recall : "+str(d[i][1]/d[i][4]))
            f.write("F-score : "+str(d[i][2]/d[i][4]))
            
# --- MAIN FUNCTION for loading cleaned dataframe and extracting features
def main():
    logger = logging.getLogger(__name__)

    # read dataframe
    logger.info('Reading data...')
    df = pd.read_csv("../../data/raw/nyt-ingredients-snapshot-2015.csv")
    
    # train data=20000, test data=2000
    df_train, df_test= train_test_split(df, train_size=5000, test_size=500, random_state=42)
    
    # train using NYTimes data
    spacy_training=convert_to_spacy(df_train)
    nlp = spacy.blank('en')  # create blank Language class
    train_spacy(nlp, spacy_training)
    
    df2 = pd.read_csv("../../data/interim/ingred_list_tagged.csv")
    
    # train using manually tagged data from merged recipes data
    spacy_training2=convert_to_spacy(df2)
    
    train_spacy(nlp, spacy_training2)
    nlp.to_disk('../../models/nlp_model')
    
    # Output test results
    spacy_test=convert_to_spacy(df_test)
    test_results(nlp, spacy_test)
    
# ----- CALLING MAIN FUNCTION
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
      
    main()