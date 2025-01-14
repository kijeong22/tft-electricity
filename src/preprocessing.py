import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

"""#### 데이터 불러오기
- train.csv: 각 건물에 대한 시간별 전력소비량 데이터 (time-series)
- building.csv: 각 건물에 대한 정보 (static)
- 위 두 데이터를 건물번호를 기준으로 병합
"""

def data_load():
    df = pd.read_csv('./dataset/train.csv')
    building = pd.read_csv('./dataset/building_info.csv')

    # # 건물 10개만 사용
    # df = df[df['건물번호'] <= 10]
    # building = building[building['건물번호'] <= 10]

    df = pd.merge(df, building, on='건물번호')

    return df


"""#### 데이터 전처리
- 결측치 처리
- 컬럼명 영문으로 변경
- TFT의 known future inputs으로 사용하기 위해 날짜를 일, 요일, 공휴일, 시간으로 분할
"""

def preprocessing(df):

    df = df.drop(columns=['num_date_time', '일조(hr)', '일사(MJ/m2)', '건물유형', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)'])
    df.columns = ['building_num', 'date', 'temperature', 'precipitation', 'windspeed', 'humidity', 'power_consumption', 'total_area', 'cooling_area']

    df.fillna(0, inplace=True)

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d %H')
    df['day'] = df.date.dt.day
    df['dayofweek'] = df.date.dt.weekday
    df['hour'] = df.date.dt.hour

    df['holiday'] = df.apply(lambda x : 0 if x['dayofweek']<5 else 1, axis = 1) # 주말을 공휴일로 간주 (관공서 등이 주말에 쉬기 때문)
    df.loc[(df.date == datetime.date(2022, 6, 6))&(df.date == datetime.date(2022, 8, 15)), 'holiday'] = 1

    df = df.drop(columns=['date'])
    df = pd.concat([df.iloc[:,0:5], df.iloc[:,6:], df.iloc[:,5:6]], axis=1)
    df = df[[
            'building_num', 'total_area', 'cooling_area', # static variables
            'day', 'dayofweek', 'hour', 'holiday', # known future variables
            'temperature', 'precipitation', 'windspeed', 'humidity', # observed past variable
            'power_consumption' # target
            ]]
    
    return df


"""#### 데이터 분할 및 스케일링
- valid와 test로 각각 일주일 데이터 사용 (24*7=168)
- StandardScaler 적용
"""

def data_split_scaling(df):

    # 데이터 분할
    # valid와 test를 24*7=168시간, 즉 일주일로 설정
    train_list = []
    valid_list = []
    test_list = []

    for i in range(len(df['building_num'].unique())):

        train, test = train_test_split(df[df['building_num'] == i+1], test_size=24*7, shuffle=False)
        train, valid = train_test_split(train[train['building_num'] == i+1], test_size=24*7, shuffle=False)
        valid = pd.concat([train[-24*7:], valid]).reset_index(drop=True) # 첫번째 값 예측을 위해서 train부분에서 입력 시퀀스 길이만큼 가져와야 함
        test = pd.concat([valid[-24*7:], test]).reset_index(drop=True) # 위와 마찬가지로 valid부분에서 입력 시퀀스 길이만큼 가져와야 함

        train_list.append(train)
        valid_list.append(valid)
        test_list.append(test)

    train_set = pd.concat(train_list).reset_index(drop=True)
    valid_set = pd.concat(valid_list).reset_index(drop=True)
    test_set = pd.concat(test_list).reset_index(drop=True)

    # 스케일링 적용
    scaler = StandardScaler()
    continuous_variables = ['total_area', 'cooling_area', 'temperature', 'precipitation', 'windspeed', 'humidity', 'power_consumption']

    train_set[continuous_variables] = scaler.fit_transform(train_set[continuous_variables])
    valid_set[continuous_variables] = scaler.transform(valid_set[continuous_variables])
    test_set[continuous_variables] = scaler.transform(test_set[continuous_variables])

    return train_set, valid_set, test_set


"""#### 데이터 로더 구축
- 데이터가 현재 건물번호를 기준으로 정렬이 되어있는데, 하나의 시퀀스 안에 서로 다른 건물번호가 있으면 안 되기 때문에 이를 고려해서 sliding window를 세밀하게 적용해야 함
- 데이터 로더의 한 개의 배치는 아래의 데이터들로 구성되어 있음
  - static_categorical_data: building_num (시간에 따라 변하지 않는 변수)
  - static_continuous_data: total_area, cooling_area (시간에 따라 변하지 않는 변수)
  - past_categorical_data: day, dayofweek, hour, holiday (관측된 과거 시점)
  - past_continuous_data: temperature, precipitation, windspeed, humidity, power_consumption (관측된 과거 시점)
  - future_data: day, dayofweek, hour, holiday (미래 시점)
  - target: power_consumption (미래 시점)
"""

class CustomDataset(Dataset):
    def __init__(self,
                 data,
                 encoder_len:int,
                 decoder_len:int,
                 variable_dict:dict,
                 stride=1
                 ):
        super(CustomDataset, self).__init__()

        self.data = data
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.stride = stride
        self.sequence_len = self.encoder_len + self.decoder_len # 192

        self.num_buildings = len(data['building_num'].unique()) # 10
        self.each_build_seq_data_len = len(self.data) // self.num_buildings - self.sequence_len + 1

        self.static_cate_data = data[variable_dict["static_categorical_variables"]]
        self.static_conti_data = data[variable_dict["static_continuous_variables"]]
        self.past_cate_data = data[variable_dict["past_categorical_variables"]]
        self.past_conti_data = data[variable_dict["past_continuous_variables"]]
        self.future_data = data[variable_dict["future_variables"]]
        self.target = data[variable_dict["target"]]

    def __len__(self):

        data_len = ((len(self.data) // self.num_buildings - self.sequence_len) // self.stride + 1) * self.num_buildings

        return data_len

    def __getitem__(self, idx):

        new_idx = idx + (idx//self.each_build_seq_data_len) * (self.sequence_len-1)

        static_cate_data = torch.tensor(self.static_cate_data[new_idx:new_idx+1].to_numpy())
        static_conti_data = torch.tensor(self.static_conti_data[new_idx:new_idx+1].to_numpy())

        past_cate_data = torch.tensor(self.past_cate_data[new_idx:new_idx+self.encoder_len].to_numpy())
        past_conti_data = torch.tensor(self.past_conti_data[new_idx:new_idx+self.encoder_len].to_numpy())

        future_data = torch.tensor(self.future_data[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())

        target = torch.tensor(self.target[new_idx+self.encoder_len:new_idx+self.sequence_len].to_numpy())

        return static_cate_data, static_conti_data, past_cate_data, past_conti_data, future_data, target


