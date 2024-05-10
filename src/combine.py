import pandas as pd

class CombineData:
    def __init__(self, *city_names) -> None:

        # TODO: 可以接無限個城市
        self.city_names = city_names

    def concat_radar_data(self):
        all_data  = []
        for city_name in self.city_names:
            city_data = pd.read_csv(fr"../output/{city_name}_radar.csv")
            all_data.append(city_data)

        combined_data = (pd.concat(all_data, axis=0, ignore_index= True)
                         .drop(['Unnamed: 0'], axis=1))
        combined_data.to_csv(fr'../output/{self.city_names}_radar.csv')

    def concat_sankey_data(self):
        all_data  = []
        for city_name in self.city_names:
            city_data = pd.read_csv(fr"../output/{city_name}_sankey.csv")
            city_data = (city_data.rename(columns={'target':f'target_{city_name}', 'city_indicator':f'{city_name}_indicator'}))
            all_data.append(city_data)

        data = pd.concat(all_data, axis=0, ignore_index=True).drop(['Unnamed: 0'], axis=1)
        data.to_csv(fr'../output/{self.city_names}_sankey.csv')

    def concat_card_data(self):
        all_data  = []
        for city_name in self.city_names:
            city_data = pd.read_csv(fr"../output/{city_name}_card.csv")
            city_data = (city_data.rename(columns={'target':f'target_{city_name}', 'city_indicator':f'{city_name}_indicator'}))
            all_data.append(city_data)

        data = (pd.concat(all_data, axis=0, ignore_index=True)
                .drop(['Unnamed: 0'], axis=1))
        
        data.to_csv(fr'../output/{self.city_names}_card.csv')
        

if __name__ == '__main__':
    city_name_1 = 'NewTaipei'
    city_name_2 = 'Taipei'
    city_name_3 = 'Taoyuan'

    combine = CombineData(city_name_1, city_name_2, city_name_3)
    combine.concat_sankey_data()
    combine.concat_card_data()
    combine.concat_radar_data()