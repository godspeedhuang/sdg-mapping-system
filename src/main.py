from sentence_transformers import SentenceTransformer

from combine import CombineData
from process import Comparison

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

sdgs_path = r'../data/ntp_sdgs_indicators.csv'
city_name = 'NewTaipei_test'
threshold = 0.5
test = Comparison(sdgs_path, city_name, model, threshold)

data = test.run()