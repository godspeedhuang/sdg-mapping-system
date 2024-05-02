import pandas as pd
from sentence_transformers import SentenceTransformer, util
import statistics
from tqdm import tqdm
tqdm.pandas()

## SDGs Color Map
SDGS_COLOR_CODE = {
    '0':'#000000',
    '1':'#E5243B',
    '2':'#DDA63A',
    '3':'#4C9F38',
    '4':'#C5192D',
    '5':'#FF3A21',
    '6':'#26BDE2',
    '7':'#FCC30B',
    '8':'#A21942',
    '9':'#FD6925',
    '10':'#DD1367',
    '11':'#FD9D24',
    '12':'#BF8B2E',
    '13':'#3F7E44',
    '14':'#0A97D9',
    '15':'#56C02B',
    '16':'#00689D',
    '17':'#19486A',
}

SDG_GOAL_TEXT = [    
    {'code':'SDG01', 'Goal':'1', 'Goal_text':'No Poverty', 'indicator':'End poverty in all its forms everywhere'},
    {'code':'SDG02', 'Goal':'2', 'Goal_text':'Zero Hunger','indicator':'End hunger, achieve food security and improved nutrition and promote sustainable agriculture'},
    {'code':'SDG03', 'Goal':'3', 'Goal_text':'Good Health and Well-Being','indicator':'Ensure healthy lives and promote well-being for all at all ages'},
    {'code':'SDG04', 'Goal':'4', 'Goal_text':'Quality Education','indicator':'Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all'},
    {'code':'SDG05', 'Goal':'5', 'Goal_text':'Gender Equality','indicator':'Achieve gender equality and empower all women and girls'},
    {'code':'SDG06', 'Goal':'6', 'Goal_text':'Clean Water and Sanitation','indicator':'Ensure availability and sustainable management of water and sanitation for all'},
    {'code':'SDG07', 'Goal':'7', 'Goal_text':'Affordable and Clean Energy','indicator':'Ensure access to affordable, reliable, sustainable and modern energy for all'},
    {'code':'SDG08', 'Goal':'8', 'Goal_text':'Decent Work and Economic Growth','indicator':'Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all'},
    {'code':'SDG09', 'Goal':'9', 'Goal_text':'Industry, Innovation and Infrastructure','indicator':'Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation'},
    {'code':'SDG10', 'Goal':'10', 'Goal_text':'Reduced Inequalities','indicator':'Reduce inequality within and among countries'},
    {'code':'SDG11', 'Goal':'11', 'Goal_text':'Sustainable Cities and Communities','indicator':'Make cities and human settlements inclusive, safe, resilient and sustainable'},
    {'code':'SDG12', 'Goal':'12', 'Goal_text':'Responsible Consumption and Production','indicator':'Ensure sustainable consumption and production patterns'},
    {'code':'SDG13', 'Goal':'13', 'Goal_text':'Climate Action','indicator':'Take urgent action to combat climate change and its impacts'},
    {'code':'SDG14', 'Goal':'14', 'Goal_text':'Life Below Water','indicator':'Conserve and sustainably use the oceans, seas and marine resources for sustainable development'},
    {'code':'SDG15', 'Goal':'15', 'Goal_text':'Life on Land','indicator':'Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss'},
    {'code':'SDG16', 'Goal':'16', 'Goal_text':'Peace, Justice and Strong Institutions','indicator':'Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels'},
    {'code':'SDG17', 'Goal':'17', 'Goal_text':'Partnerships for the Goals','indicator':'Strengthen the means of implementation and revitalize the Global Partnership for Sustainable Development'},
]

world_bank_custodian_indicators = [
    'Indicator1-1-1',
    'Indicator1-2-1',
    'Indicator1-4-2',
    'Indicator3-8-2',
    'Indicator5-1-1',
    'Indicator7-1-1',
    'Indicator8-10-2',
    'Indicator9-1-1',
    'Indicator9-3-2',
    'Indicator10-1-1',
    'Indicator10-2-1',
    'Indicator10-4-2',
    'Indicator10-7-1',
    'Indicator10-c-1',
    'Indicator15-a-1',
    'Indicator15-b-1',
    'Indicator16-5-2',
    'Indicator16-6-1',
    'Indicator17-3-2',
    'Indicator17-4-1',
    'Indicator17-13-1',
    'Indicator17-17-1',
]

world_bank_involved_indicators = [
    'Indicator1-2-2',
    'Indicator1-3-1',
    'Indicator2-3-2',
    'Indicator3-1-1',
    'Indicator3-2-1',
    'Indicator3-2-2',
    'Indicator4-2-1',
    'Indicator4-6-1',
    'Indicator5-5-1',
    'Indicator5-a-1',
    'Indicator5-a-2',
    'Indicator7-2-1',
    'Indicator7-3-1',
    'Indicator8-1-1',
    'Indicator8-2-1',
    'Indicator8-b-1',
    'Indicator9-2-1',
    'Indicator10-7-2',
    'Indicator16-10-2',
    'Indicator17-1-1',
    'Indicator17-18-3',
    'Indicator17-19-1',
]


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
    
    def compute_similarity_by_divided_sentence(self, length):
        # un_indicator = self.un_sdgs['indicator']
        # city_sentences = self.city_sdgs['sentence'].to_list()
        

        def _segment_cosine_similarity(word_list, city_sentence):
            sentence = ' '.join(word_list)
            embedding_sentence = self.model.encode(sentence, convert_to_tensor = True)
            # embedding_city = self.model.encode(self.city_sdgs)
            
            # TODO: How to import city goal
            embedding_city = self.model.encode(city_sentence, convert_to_tensor = True)

            cosine_score = util.cos_sim(embedding_city, embedding_sentence)
            return cosine_score.cpu().item()

        def _split_sentence(text):
            score_list = []
            word = text.split()
            word_count = len(word)
            each_seg_word_count = int(round(word_count/length, 0))
            
            # How to implement
            # for loop for city sdg
            # for city_sentence in self.city_sdgs['sentence'].to_list():
            city_sentence = '( 低收入戶人數 ÷ 新北市總人數）× 100%'

            for id, i in enumerate(range(0, word_count, each_seg_word_count)):
                score = _segment_cosine_similarity(word[i:i+each_seg_word_count], city_sentence)
                score_list.extend([score])

            average_cosine = statistics.mean(score_list)
            return average_cosine

        un_description_segs = self.un_sdgs.progress_apply(lambda row: _split_sentence(row['sentence']), axis=1)
        
        pd.concat([self.un_sdgs, un_description_segs], axis=1).to_csv('test3.csv')

        # city_sentences = self.city_sdgs['sentence'].to_list()

    
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

        # Create the relationship with World Bank
        world_bank_custodian_indicators
        self.df_filter.loc[self.df_filter['source'].isin(world_bank_custodian_indicators), 'WB_related'] = 'world bank custodian'
        self.df_filter.loc[self.df_filter['source'].isin(world_bank_involved_indicators), 'WB_related'] = 'world bank involved'
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
                    .drop(['code'], axis=1))
        
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
        self.compute_similarity_by_divided_sentence(10)


        # process
        # df = self.compute_similarity()
        # self.melt_table(df)

        # self.fileter_by_threshold()
        # self.export_radar_chart_data()
        # self.export_card_data()
        # self.export_sankey_chart_data()




if __name__ == '__main__':

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    sdgs_path = r'../data/ntp_sdgs_indicators.csv'
    city_name = 'Testing'
    threshold = 0.5
    test = Comparison(sdgs_path, city_name, model, threshold)
    
    # sdgs_path = r'../data/tp_sdgs_indicators.csv'
    # city_name = 'Taipei'
    # test = Comparison(sdgs_path, city_name, model)
    
    data = test.run()
    # data.to_csv('../output/test.csv')
