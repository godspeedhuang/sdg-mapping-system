import pandas as pd

class CombineData:
    def __init__(self, city_name_1, city_name_2) -> None:

        # TODO: 可以接無限個城市
        self.city_name_1 = city_name_1
        self.city_name_2 = city_name_2

    def concat_radar_data(self):
        city_1 = pd.read_csv(fr"../output/{self.city_name_1}_radar.csv")
        city_2 = pd.read_csv(fr"../output/{self.city_name_2}_radar.csv")

        data = (pd.concat([city_1, city_2], axis=0, ignore_index=True)
                .drop(['Unnamed: 0'], axis=1))
        data.to_csv(fr'../output/{self.city_name_1}_{self.city_name_2}_radar.csv')

    def concat_sankey_data(self):
        city_1 = pd.read_csv(fr"../output/{self.city_name_1}_sankey.csv")
        city_2 = pd.read_csv(fr"../output/{self.city_name_2}_sankey.csv")

        city_1 = (city_1.rename(columns={'target':f'target_{self.city_name_1}', 'city_indicator':f'{self.city_name_1}_indicator'}))
        city_2 = (city_2.rename(columns={'target':f'target_{self.city_name_2}', 'city_indicator':f'{self.city_name_2}_indicator'}))

        data = pd.concat([city_1, city_2], axis=0, ignore_index=True).drop(['Unnamed: 0'], axis=1)
        data.to_csv(fr'../output/{self.city_name_1}_{self.city_name_2}_sankey.csv')

    def concat_card_data(self):
        city_1 = pd.read_csv(fr"../output/{self.city_name_1}_card.csv")
        city_2 = pd.read_csv(fr"../output/{self.city_name_2}_card.csv")

        data = (pd.concat([city_1, city_2], axis=0, ignore_index=True)
                .drop(['Unnamed: 0'], axis=1))
        
        data.to_csv(fr'../output/{self.city_name_1}_{self.city_name_2}_card.csv')
        

if __name__ == '__main__':
    city_name_1 = 'NewTaipei'
    city_name_2 = 'Taipei'

    combine = CombineData(city_name_1, city_name_2)
    combine.concat_sankey_data()