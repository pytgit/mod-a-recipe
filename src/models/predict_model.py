import pandas as pd
import logging

def suggest_ingred(ingreds, recipes_ingred):
    init_ingreds=set(ingreds)
    #print("selected recipe:", init_ingreds)
    suggest=[]
    for item in recipes_ingred:
        diff = [x for x in item[1] if x not in init_ingreds]
        #print('diff: ', diff)
        suggest.append([item[0], diff])
    return suggest

# --- MAIN FUNCTION for loading cleaned dataframe and extracting features
def main():
    logger = logging.getLogger(__name__)

    # read dataframe
    logger.info('Get recipe index...')
    df=pd.read_pickle('../../data/processed/sim_recipes_list_df.pkl')
    
    INDEX=3 # fake an index for now
    # get similar recipes and their ingredient list
    init_list=df.iloc[INDEX].ingred_s
    sim_list=df.iloc[INDEX].sim_recipes
    sim_list_ingreds=list(df.iloc[sim_list].ingred_s)
    sim_pairs=zip(sim_list, sim_list_ingreds)
    
    # get suggested ingredients per suggested recipes
    suggestion=suggest_ingred(init_list,sim_pairs)
    print(suggestion)
    
# ----- CALLING MAIN FUNCTION
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
      
    main()