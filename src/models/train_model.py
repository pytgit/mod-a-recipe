import pandas as pd
import numpy as np
import string
import pickle
import logging
import spacy
import datetime

# NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

# to track time
from time import time

def split_ingred(ingredients):
    ingredients=[x for x in ingredients if len(x)>2]
    return ingredients

def save_NMF_data(model, feature_names, no_top_words, topic_names):
    filepath='../../models/nmf_topics_'+str(no_top_words)+'_'+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))+'.txt'
    with open(filepath, 'w') as f:
        f.write('NMF reconstruction error is: '+str(model.reconstruction_err_))
    
        for ix, topic in enumerate(model.components_):
            f.write("\n'"+topic_names[ix]+"' : ")
            f.write(", ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

# calculate cosine similarity while breaking up computations
def cosine_similarity_top_n(m1, threshold=0.9):
    
    ret = np.empty((m1.shape[0],),dtype=object)
    #print(ret.shape)
    for i in range(0, m1.shape[0]):
        vec=cosine_similarity(m1[i,:].reshape(1, -1), m1[:,:])
        v_vec={i:v for i, v in enumerate(vec[0]) if v>threshold}
        sorted_d = sorted(v_vec.items(), key=lambda x: x[1], reverse=True)
        sort_vec=[key for key, value in sorted_d]
        ret[i]=sort_vec[1:]
        if i%10000==0:
            print('Cosine similarity iterations: at row '+str(i)+' out of '+str(m1.shape[0]))
        
    return ret

# MAIN FUNCTION for loading cleaned dataframe and extracting features
def main():
    logger = logging.getLogger(__name__)

    # read dataframe
    logger.info('Reading data...')
    df=pd.read_pickle('../../data/processed/df_for_model_spacy.pkl')
    
    # create TFIDF vector
    logger.info('Creating TFIDF vector for recipes-to-ingredients...')
    #ingred_s_list=np.unique(df.ingred_s.sum())
    tfidf_vectorizer = TfidfVectorizer(tokenizer=split_ingred, lowercase=False)
    tfidf_doc_ingred = tfidf_vectorizer.fit_transform(df.ingred_s)

    # run NMF
    logger.info('Running Non-negative matrix factorization topic modeling...')
    NUM_TOPICS=50 # hyper parameter that can be tweaked for NMF topic modeling
    NUM_TOP_WORDS=10 # to save to file for NMF model
    
    topic_names=['recipe'+str(i) for i in range(0, NUM_TOPICS)]
    nmf_model = NMF(NUM_TOPICS)
    nmf_doc_topic = nmf_model.fit_transform(tfidf_doc_ingred)
    
    # save relevant NMF info
    logger.info('Saving NMF topic model and data...')
    pickle.dump(nmf_doc_topic, open('../../data/interim/nmf_doc_topic.pkl', 'wb'))

    save_NMF_data(nmf_model, tfidf_vectorizer.get_feature_names(), NUM_TOP_WORDS, topic_names)
    
    # calculate consine similarity of recipes
    logger.info('Calculate cosine similarity...')
    THRESHOLD=0.9
    sim_recipes=cosine_similarity_top_n(nmf_doc_topic, THRESHOLD)
    sim_recipes_df=pd.DataFrame(sim_recipes)
    sim_recipes_df.to_pickle('../../data/interim/sim_recipes_df.pkl')
    
    # get list of similar recipes per recipes
    logger.info('Extract list of similar recipes...')
    df=df.reset_index()
    df.loc[:,'sim_recipes_list']=pd.Series(sim_recipes_df[0].values, index=sim_recipes_df.index)
    sim_recipes_df.to_pickle('../../data/processed/sim_recipes_list_df.pkl')
    
# CALLING MAIN FUNCTION
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
      
    main()