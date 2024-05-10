import pandas as pd
from sentence_transformers import SentenceTransformer, util
import statistics
from tqdm import tqdm
import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize

from constants import (
    SDGS_COLOR_CODE,
    SDG_GOAL_TEXT,
    WORLD_BANK_CUSTODIAN_INDICATORS,
    WORLD_BANK_INVOLVED_INDICATORS
)

tqdm.pandas()


os.environ['TOKENIZERS_PARALLELISM'] = 'true'

class Comparison():

    def __init__(self,
                  city_file, 
                  city_name,
                  model,
                  threshold=0.5,
                  ):

        # import UN indicators
        un_file = r'../data/un_sdgs_indicators.csv'
        self.un_sdgs = pd.read_csv(un_file)

        # import city indicators
        self.city_sdgs = pd.read_csv(city_file)
        self.city_sdgs = self.city_sdgs[['goal','id','indicator','description']]
        # .rename(columns={'indicator':'indicator', 'description':'description'})

        # select a model for creating embeddings
        self.model = model
        self.city_name = city_name
        self.threshold = threshold

        # Initialize empty dataframe
        self.df_melt = pd.DataFrame()
        self.df_filter = pd.DataFrame()

    def un_sdgs_preprocess(self):
        self.un_sdgs['sentence'] = self.un_sdgs['indicator'] + ": "+ self.un_sdgs['description']
        pattern = r'Indicator(?P<Goal>\d+)-'
        
        # create goal column
        self.un_sdgs.insert(2, 'goal', self.un_sdgs['id'].str.extract(pattern))
        self.un_sdgs.drop([self.un_sdgs.columns[0]],axis=1,inplace=True) 

        # mapping color column
        self.un_sdgs['color'] = self.un_sdgs['goal'].map(SDGS_COLOR_CODE)

    def city_sgds_preprocess(self):
        # self.city_sdgs['sentence'] = self.city_sdgs['indicator'] + ": " + self.city_sdgs['description'].fillna('')
        
        self.city_sdgs['sentence'] = self.city_sdgs['indicator'] + ": " + self.city_sdgs['description'].fillna('')
        

        # convert goal and id columns to string
        self.city_sdgs = self.city_sdgs.astype('str')

    def compute_similarity(self):
        un_sentences = self.un_sdgs['sentence'].to_list()
        city_sentences = self.city_sdgs['sentence'].to_list()

        embeddings_un = self.model.encode(un_sentences, convert_to_tensor=True)
        embeddings_city = self.model.encode(city_sentences, convert_to_tensor=True)

        cosine_scores = util.cos_sim(embeddings_un, embeddings_city)
        
        # create mapping table (nxm)
        un_id = self.un_sdgs['id'].to_list()
        city_id = self.city_sdgs['id'].to_list()
        df = pd.DataFrame(cosine_scores.cpu(), columns=city_id, index=un_id)

        return df
    
    def compute_similarity_by_divided_sentence(self):
        """Split sentences by two different ways"""

        def package_initialize():
            nltk.download('punkt')

        def _evenly_split_un_description(un_description, length):
            word_list = un_description.split(" ") # split by blank space
            word_count = len(word_list)
            each_seg_word_count = int(round(word_count/length, 0)) # how many words in each segment
            seg_list = []
            for i in range(0, word_count, each_seg_word_count):
                seg_sentence = ' '.join(word_list[i:i+each_seg_word_count])
                seg_list.append(seg_sentence)
            return seg_list

        def _sentence_split_un_description_spacy(un_description):
            """Use by Spacy"""
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(un_description)
            sentences = [sentence for sentence in doc.sents]
            return sentences
        
        def _sentence_split_un_description_nltk(un_description):
            """Use by NLTK"""
            return sent_tokenize(un_description)

        # # create the embedding of the city SDGs
        embedding_cities = self.model.encode(self.city_sdgs['sentence'].to_list(), convert_to_tensor=True) 

        # compute similarity for indicator
        embedding_un_indicator = self.model.encode(self.un_sdgs['indicator'].to_list(), convert_to_tensor=True)
        cosine_scores_indicator = util.cos_sim(embedding_un_indicator, embedding_cities)

        cosine_scores_indicator = pd.DataFrame(cosine_scores_indicator.cpu(), columns=self.city_sdgs['id'].to_list())
        indicator_similarity = pd.concat([self.un_sdgs, cosine_scores_indicator], axis=1)

        # by evenly split
        # by spacy
        # self.un_sdgs['segment'] = self.un_sdgs['description'].apply(_evenly_split_un_description_spacy)
        # by NLTK
        self.un_sdgs['segment'] = self.un_sdgs['description'].apply(_sentence_split_un_description_nltk)
        un_sdgs_explode = self.un_sdgs.explode('segment', ignore_index=True)
        
        # create embeddings for each sentence
        embedding_un_segment = self.model.encode(un_sdgs_explode['segment'].to_list(), convert_to_tensor=True) 

        # # compute_similarity
        cosine_scores = util.cos_sim(embedding_un_segment, embedding_cities)
        cosine_scores = pd.DataFrame(cosine_scores.cpu(), columns=self.city_sdgs['id'].to_list())
        
        similarity_df = pd.concat([un_sdgs_explode, cosine_scores], axis=1)
        indicator_similarity['segment'] = ''

        similarity_df = pd.concat([similarity_df, indicator_similarity], axis=0, ignore_index=True)  

        # similarity_avg = similarity_df.groupby(['id'])['1':'81'].mean()
        similarity_col = similarity_df.columns[-self.city_sdgs.shape[0]:]
        similarity_avg = similarity_df.groupby(['id'])[similarity_col].mean()

        # match the self.melt_table method
        similarity_avg.index.names = ['index']
        return similarity_avg

    
    def melt_table(self, df):
        # index: un_id
        self.df_melt = df.reset_index()
        self.df_melt = pd.melt(self.df_melt, id_vars=['index'], value_name = 'value')
        self.df_melt = self.df_melt.rename(columns={'index':'source', 'variable':'target', 'value':'value'})
        # return self.df_melt
    
    def fileter_by_threshold(self):
        # Keep the mapping pairs above than the threshold
        self.df_filter = self.df_melt.loc[self.df_melt['value']>self.threshold, :]

        # select the indicators of UN that aren't mapped by the city indicators
        not_mapped_id = set(self.un_sdgs['id'].to_list()) - set(self.df_filter['source'].unique())

        # assign those not mapped indicators for target 'Nothing'
        for source in not_mapped_id:
            self.df_filter = pd.concat([self.df_filter, 
                                          pd.Series({'source':source, 'target':'Nothing', 'value':0.01}).to_frame().T],
                                            axis=0, ignore_index=True)
        
        # Create a row 'None' to city_sdgs
        not_mapped_record = {
            'id':'Nothing',
            'goal':'Nothing',
            'indicator':'Nothing',
        }
        city_sdgs_w_nothing = pd.concat([self.city_sdgs, pd.Series(not_mapped_record).to_frame().T], axis=0, ignore_index=True)

        # merge with indicator text from target(un) and source(city)
        self.df_filter = (self.df_filter
                          .merge(self.un_sdgs[['id', 'goal', 'indicator']], left_on='source', right_on='id', how='inner')
                          .drop(['id'], axis=1)
                          .rename(columns={'indicator':'un_indicator'})
                          .merge(city_sdgs_w_nothing[['id', 'indicator']], left_on='target', right_on='id', how='inner')
                          .drop(['id'], axis=1)
                          .rename(columns={'indicator':'city_indicator'}))
        
        # goal to SDG goal SDG01, SDG02, SDG03
        self.df_filter['goal'] = self.df_filter['goal'].apply(lambda x:'SDG{:02d}'.format(int(x)))
        self.df_filter = (self.df_filter.merge(pd.DataFrame(SDG_GOAL_TEXT)[['code', 'Goal_text']], left_on='goal', right_on='code', how='inner')
                          .drop(['code'], axis=1)
                          .rename(columns={'Goal_text':'goal_text'}))

        # Create the relationship with World Bank
        self.df_filter.loc[self.df_filter['source'].isin(WORLD_BANK_CUSTODIAN_INDICATORS), 'WB_related'] = 'world bank custodian'
        self.df_filter.loc[self.df_filter['source'].isin(WORLD_BANK_INVOLVED_INDICATORS), 'WB_related'] = 'world bank involved'
        self.df_filter.loc[self.df_filter['WB_related'].isna(), 'WB_related'] = 'non-related'

        # set a column to filter paired or unpaired
        self.df_filter['paired'] = self.df_filter.apply(lambda x: 'unpaired' if x['target']=='Nothing' else 'paired', axis=1)


    def export_sankey_chart_data(self):
        self.df_filter.to_csv(fr'../output/{self.city_name}_sankey.csv')

    def export_card_data(self):
        # Count non-None indicators
        # print(set(self.un_sdgs['id'].to_list()))
        # print(set(self.df_filter['source'].unique()))
        
        # How many UN indicators mapped by local indicators
        not_mapped_id = self.un_sdgs.shape[0]- self.df_filter[self.df_filter['target']!='Nothing']['source'].nunique()
        mapped_id = self.un_sdgs.shape[0]- not_mapped_id

        # How many local indicators mapped to UN indicators
        mapped_city_id = self.df_filter['target'].nunique()
        
        card_data = [{
            'city_name': self.city_name,
            'un_coverage': f"{mapped_id}/{self.un_sdgs['id'].nunique()} ({round((mapped_id/ self.un_sdgs.shape[0])*100, 2)}%)",
            # 'covered_un_percentage': f"{round((mapped_id/ self.un_sdgs.shape[0])*100, 2)}%",
            'mapped_indicator':f"{mapped_city_id}/{self.city_sdgs.shape[0]} ({round((mapped_city_id/self.city_sdgs['id'].nunique())*100, 2)}%)",
        }]

        pd.DataFrame(card_data).to_csv(fr'../output/{self.city_name}_card.csv')

    def export_radar_chart_data(self):
        
        # drop duplicate un indicators
        unique_un_sdg = self.df_filter.drop_duplicates(subset=['source'])
        city_unique_un_id = unique_un_sdg[unique_un_sdg['target']!='Nothing']
        city_unique_un_id = city_unique_un_id.groupby('goal', group_keys=False)['target'].apply(lambda x:(x!='Nothing').sum())
        
        unique_un_id = unique_un_sdg['goal'].value_counts().sort_index()
        
        radar_df = ((city_unique_un_id/unique_un_id)
                    .round(2)
                    .reset_index()
                    .rename(columns={0:'value'})
                    .fillna(0)
                    .merge(pd.DataFrame(SDG_GOAL_TEXT)[['code','Goal_text']], left_on='goal', right_on='code', how='inner')
                    .rename(columns={'Goal_text':'goal_text'})
                    .drop(['code'], axis=1))
        
        # SDG01 No Poverty
        radar_df['goal_text'] = radar_df['goal']+ '\n' +radar_df['goal_text']

        # sort by goal number SDG1-> SDG2 -> ... -> SDG17
        radar_df['goal_num'] = radar_df['goal'].str[3:]
        radar_df['goal_num'] = radar_df['goal_num'].astype('int')
        radar_df = (radar_df.sort_values(by='goal_num', ascending=True)
                            .reset_index()
                            .drop(['index'], axis=1))
        
        radar_df.insert(0, 'cityname', self.city_name)
        
        radar_df.to_csv(fr'../output/{self.city_name}_radar.csv')
        # test = test.groupby('Goal', group_keys=False)['target'].apply(lambda x:(x!='0').sum())
        # total_counts = df_melt_filtered_merged['Goal'].value_counts().sort_index()

    def run(self):
        
        # preprocess
        self.un_sdgs_preprocess()
        self.city_sgds_preprocess()

        # def test_func():
        #     self.compute_similarity_by_divided_sentence()

        # t = timeit.timeit(test_func, number=1)
        # print("執行時間：%f 秒" % t)
        

        
        # process
        df = self.compute_similarity()
        # df = self.compute_similarity_by_divided_sentence()
        self.melt_table(df)
        
        self.fileter_by_threshold()
        self.export_radar_chart_data()
        self.export_card_data()
        self.export_sankey_chart_data()




if __name__ == '__main__':

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    sdgs_path = r'../data/ty_sdgs_indicators.csv'
    city_name = 'Taoyuan'
    threshold = 0.5
    test = Comparison(sdgs_path, city_name, model, threshold)
    
    # sdgs_path = r'../data/tp_sdgs_indicators.csv'
    # city_name = 'Taipei'
    # test = Comparison(sdgs_path, city_name, model)
    
    data = test.run()
    # data.to_csv('../output/test.csv')
