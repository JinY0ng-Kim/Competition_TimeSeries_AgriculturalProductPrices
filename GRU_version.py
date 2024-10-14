import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import random

# 랜덤 시드 설정
seed = 42
torch.manual_seed(seed)  # PyTorch 랜덤 시드
np.random.seed(seed)      # NumPy 랜덤 시드
random.seed(seed)         # Python 기본 랜덤 시드

config = {
    "learning_rate": 1e-5,
    "epoch": 10,
    "batch_size": 1024,
    "output_size": 3
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_file = './logs_128h_9l_9s.txt'

CFG = SimpleNamespace(**config)

품목_리스트 = ['건고추', '사과', '감자', '배', '깐마늘(국산)', '무', '상추', '배추', '양파', '대파']


def process_data(raw_file, 산지공판장_file, 전국도매_file, 품목명, scaler=None):
    raw_data = pd.read_csv(raw_file)
    산지공판장 = pd.read_csv(산지공판장_file)
    전국도매 = pd.read_csv(전국도매_file)

    # 변수 초기화
    공판장_데이터 = None
    도매_데이터 = None

    # 타겟 및 메타데이터 필터 조건 정의
    conditions = {
    '감자': {
        'target': lambda df: (df['품종명'] == '감자 수미') & (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': None,
        '도매': {'시장명': ['*전국도매시장', '강릉', '광주각화', '광주서부','대전오정', 
                            '서울가락','울산', '인천남촌', '인천삼산', '천안', 
                            '충주' ], '품목명': ['감자'], '품종명': ['수미']}
    },
    '건고추': {
        'target': lambda df: (df['품종명'] == '화건') & (df['거래단위'] == '30 kg') & (df['등급'] == '상품'),
        '공판장': None, 
        '도매': None  
    },
    '깐마늘(국산)': {
        'target': lambda df: (df['거래단위'] == '20 kg') & (df['등급'] == '상품'),
        '공판장': None,
        '도매': {'시장명': ['*전국도매시장', '익산', '정읍'], 
                                    '품목명': ['마늘'], '품종명': ['깐마늘']}
    },
    '대파': {
        'target': lambda df: (df['품종명'] == '대파(일반)') & (df['거래단위'] == '1키로단') & (df['등급'] == '상'),
        '공판장': None,
        '도매': {'시장명': ['*전국도매시장', '강릉', '광주각화', '광주서부', '구리',
                            '구미', '대구북부', '대전노은', '대전오정', '부산반여', 
                            '부산엄궁', '서울가락', '서울강서', '수원', '순천', 
                            '안산', '안양', '울산', '원주', '인천남촌',
                            '인천삼산', '전주', '정읍', '진주', '창원내서',
                            '창원팔용', '천안', '청주', '춘천', '충주',
                            '포항'], '품목명': ['대파'], '품종명': ['대파(일반)']}
    },
    '무': {
        'target': lambda df: (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['무'], '품종명': ['기타무'], '등급명': ['상']},
        '도매': None
    },
    '배추': {
        'target': lambda df: (df['거래단위'] == '10키로망대') & (df['등급'] == '상'),
        '공판장': None,
        '도매': None
    },
    '사과': {
        'target': lambda df: (df['품종명'].isin(['홍로', '후지'])) & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
        '공판장': None,
        '도매': {'시장명': ['*전국도매시장', '광주각화', '서울가락'], '품목명': ['사과'], '품종명': ['후지']}
    },
    '상추': {
        'target': lambda df: (df['품종명'] == '청') & (df['거래단위'] == '100 g') & (df['등급'] == '상품'),
        '공판장': None,
        '도매': {'시장명': ['*전국도매시장', '강릉', '광주각화', '광주서부', '구리', 
                            '대구북부', '대전노은', '대전오정', '부산반여', '부산엄궁',
                            '서울가락', '서울강서', '수원', '순천', '안산', 
                            '안양', '울산', '원주', '인천남촌', '인천삼산', 
                            '전주', '정읍', '진주', '창원내서', '창원팔용', 
                            '천안', '청주', '충주'], '품목명': ['상추'], '품종명': ['청상추']}
    },
    '양파': {
        'target': lambda df: (df['품종명'] == '양파') & (df['거래단위'] == '1키로') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['양파'], '품종명': ['기타양파'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장', '강릉', '대구북부', '대전노은', '대전오정', 
                            '수원', '울산', '인천남촌', '인천삼산', '진주'], '품목명': ['양파'], '품종명': ['양파(일반)']}
    },
    '배': {
        'target': lambda df: (df['품종명'] == '신고') & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['배'], '품종명': ['신고'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장', '광주각화', '대구북부', '부산반여', '부산엄궁', 
                            '서울가락', '진주'], '품목명': ['배'], '품종명': ['신고']}
    }
    }

    # 타겟 데이터 필터링
    raw_품목 = raw_data[raw_data['품목명'] == 품목명]
    target_mask = conditions[품목명]['target'](raw_품목)
    filtered_data = raw_품목[target_mask]

    # 필요한 열만 선택
    filtered_data = filtered_data[['시점', '평균가격(원)']]

    # 다른 품종에 대한 파생변수 생성
    # other_data = raw_품목[~target_mask]
    # unique_combinations = other_data[['품종명', '거래단위', '등급']].drop_duplicates()
    # for _, row in unique_combinations.iterrows():
    #     품종명, 거래단위, 등급 = row['품종명'], row['거래단위'], row['등급']
    #     mask = (other_data['품종명'] == 품종명) & (other_data['거래단위'] == 거래단위) & (other_data['등급'] == 등급)
    #     temp_df = other_data[mask]
    #     for col in ['평년 평균가격(원)', '평균가격(원)']:
    #         new_col_name = f'{품종명}_{거래단위}_{등급}_{col}'
    #         filtered_data = filtered_data.merge(temp_df[['시점', col]], on='시점', how='left', suffixes=('', f'_{new_col_name}'))
    #         filtered_data.rename(columns={f'{col}_{new_col_name}': new_col_name}, inplace=True)

    단위 = {
    '감자': 20,
    '건고추': 30,
    '사과': 10,
    '배': 10,
    '깐마늘(국산)': 20,
    '무': 20,
    '상추': 0.1,
    '배추': 10,
    '양파': 1,
    '대파': 1
    }

    # 공판장 데이터 처리
    if conditions[품목명]['공판장']:
        filtered_공판장 = 산지공판장
        for key, value in conditions[품목명]['공판장'].items():
            if value is not None:
                filtered_공판장 = filtered_공판장[filtered_공판장[key].isin(value)]

        # 공판장 데이터 처리
        공판장_데이터 = filtered_공판장.pivot_table(
            values='평균가(원/kg)', 
            index='시점', 
            columns='공판장명', 
            aggfunc='first'
        ).reset_index()

        # 열 이름 변경
        공판장_열_이름 = [f'{col}_공판장_평균가격' for col in 공판장_데이터.columns if col != '시점']
        공판장_데이터.columns = ['시점'] + 공판장_열_이름

        # 단위 변환
        for col in 공판장_데이터.columns:
            if '_공판장_평균가격' in col:
                공판장_데이터[col] = 공판장_데이터[col] * 단위.get(품목명, 1)
        
        # # 결측값을 0으로 채우기
        # 공판장_데이터 = 공판장_데이터.fillna(0)

        공판장_데이터.to_csv(f'pre_normalized_data_{품목명}_with_공판장.csv', index=False, encoding='utf-8-sig')

    # 도매 데이터 처리
    if conditions[품목명]['도매']:
        filtered_도매 = 전국도매
        for key, value in conditions[품목명]['도매'].items():
            if value is not None:
                filtered_도매 = filtered_도매[filtered_도매[key].isin(value)]
        
        # 도매 데이터 처리
        도매_데이터 = filtered_도매.pivot_table(
            values='평균가(원/kg)', 
            index='시점', 
            columns='시장명', 
            aggfunc='first'
        ).reset_index()

        # 열 이름 변경
        도매_열_이름 = [f'{col}_도매_평균가격' for col in 도매_데이터.columns if col != '시점']
        도매_데이터.columns = ['시점'] + 도매_열_이름

        # 단위 변환
        for col in 도매_데이터.columns:
            if '_도매_평균가격' in col:
                도매_데이터[col] = 도매_데이터[col] * 단위.get(품목명, 1)

        # # 결측값을 0으로 채우기
        # 도매_데이터 = 도매_데이터.fillna(0)

        도매_데이터.to_csv(f'pre_normalized_data_{품목명}_with_도매.csv', index=False, encoding='utf-8-sig')

    # 모든 데이터 병합
    filtered_data_최종 = filtered_data[['시점', '평균가격(원)']]
    if 공판장_데이터 is not None:
        filtered_data_최종 = pd.merge(filtered_data_최종, 공판장_데이터, on='시점', how='outer')
    if 도매_데이터 is not None:
        filtered_data_최종 = pd.merge(filtered_data_최종, 도매_데이터, on='시점', how='outer')

    # 수치형 컬럼 처리
    numeric_columns = [col for col in filtered_data_최종.columns if '평균가격' in col]
    # filtered_data_최종[numeric_columns] = filtered_data_최종[numeric_columns].fillna(0)
    filtered_data_최종.to_csv(f'pre_normalized_data_{품목명}_with_최종.csv', index=False, encoding='utf-8-sig')

    # 정규화 적용
    if scaler is None:
        scaler = MinMaxScaler()
        filtered_data_최종[numeric_columns] = scaler.fit_transform(filtered_data_최종[numeric_columns])
    else:
        filtered_data_최종[numeric_columns] = scaler.transform(filtered_data_최종[numeric_columns])

    return filtered_data_최종, scaler
    


class AgriculturePriceDataset(Dataset):
    def __init__(self, dataframe, window_size=9, prediction_length=3, is_test=False):
        self.data = dataframe
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.is_test = is_test
        
        self.price_column = [col for col in self.data.columns if '평균가격(원)' in col and len(col.split('_')) == 1][0]
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.sequences = []
        if not self.is_test:
            for i in range(len(self.data) - self.window_size - self.prediction_length + 1):
                x = self.data[self.numeric_columns].iloc[i:i+self.window_size].values
                y = self.data[self.price_column].iloc[i+self.window_size:i+self.window_size+self.prediction_length].values
                self.sequences.append((x, y))
        else:
            self.sequences = [self.data[self.numeric_columns].values]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if not self.is_test:
            x, y = self.sequences[idx]
            return torch.FloatTensor(x), torch.FloatTensor(y)
        else:
            return torch.FloatTensor(self.sequences[idx])
        
"""
# 포지셔널 인코딩 클래스 정의
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # 포지셔널 인코딩 추가
        return x

class Att(nn.Module):
    def __init__(self, d_model = 180, num_heads=9, max_len=5000):
        super(Att, self).__init__()
        out_channels=180
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=180, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=180, out_channels=360, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=360, out_channels=720, kernel_size=1, stride=1),
            nn.BatchNorm1d(out_channels*2*2),
            nn.ReLU(inplace=True)
        )

        self.embedding = nn.Linear(720, d_model)  # Learnable Embedding (1차원 -> d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)  # 포지셔널 인코딩
        self.attention_x = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.attention_x1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.attention_x2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        
        # self.lstm = nn.LSTM(d_model, 90, 9, batch_first=True)
        
        self.fc = nn.Linear(180, 3)

    def forward(self, x):
        # h0 = torch.zeros(9, x.size(0), 90).to(x.device)
        # c0 = torch.zeros(9, x.size(0), 90).to(x.device)

        x = self.conv(x)
        
        x = x.permute(0, 2, 1)
        x = self.embedding(x)  # Learnable Embedding으로 차원 확장: [batch_size, time_steps, d_model]
        x = self.pos_encoder(x)  # 포지셔널 인코딩 추가: [batch_size, time_steps, d_model]

        x = x.permute(1, 0, 2)
        x, _ = self.attention_x(x,x,x)
        x, _ = self.attention_x1(x,x,x)
        x, _ = self.attention_x2(x,x,x)

        x = x.permute(1, 0, 2)
        x = x.contiguous().view(x.size(0), -1)

        # x, _ = self.lstm(x, (h0, c0))
        # x = x[:, -1, :]

        x = self.fc(x)
        return x
    
"""


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, segment_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear = nn.Linear(segment_size, hidden_size)  # Linear layer
        self.relu = nn.ReLU()  # ReLU activation
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 시퀀스를 세그먼트 크기로 나누기
        segments = x.view(batch_size, -1, segment_size)  # (batch_size, num_segments, segment_size)
        
        # Linear와 ReLU 적용
        segments = self.linear(segments)  # (batch_size, num_segments, hidden_size)
        segments = self.relu(segments)

        h0 = torch.zeros(self.num_layers, segments.size(0), self.hidden_size).to(x.device)  # 초기 hidden state
        out, hidden = self.gru(segments, h0)  # out: (batch_size, num_segments, hidden_size)
        return out, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, segment_size)
        self.fc3 = nn.Linear(segment_size, output_size)
        self.dropout = nn.Dropout(dropout_p)  # Dropout 추가
        #self.pos_encoder = PositionalEncoding(hidden_size)  # Position Encoding 추가

    def forward(self, x, hidden):
        #x = self.pos_encoder(x)  # Position Encoding 추가
        x = x.unsqueeze(1).repeat(1,106,1)
        # print(f"Decoder_x: {x.shape}\nhidden: {hidden.shape}")

        out, hidden = self.gru(x, hidden)
        out = self.dropout(out)  # Dropout 적용
        out = self.fc(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, segment_size):
        super(Seq2Seq, self).__init__()
        self.segment_size = segment_size
        self.encoder = Encoder(input_size, hidden_size, num_layers, segment_size)
        self.decoder = Decoder(hidden_size, output_size, num_layers)
        
    def segment_data(self, x):
        batch_size, seq_len, input_size = x.size()
        num_segments = seq_len // self.segment_size
        x = x[:, :num_segments * self.segment_size, :]
        x = x.view(batch_size, num_segments, self.segment_size, input_size)
        return x
    
    def forward(self, x):
        segmented_data = self.segment_data(x)
        batch_size, num_segments, segment_size, input_size = segmented_data.size()
        
        encoded_segments = []
        for i in range(num_segments):
            segment = segmented_data[:, i, :, :]
            out, hidden = self.encoder(segment)
            # print("encoded_segments: ",out.shape)
            encoded_segments.append(out[:, -1, :])

        encoded_segments = torch.cat(encoded_segments, dim=0)

        # print("cat_egments: ",encoded_segments.shape)
        
        # Decoder로 전달 (위치 인코딩 및 Dropout 적용)
        decoder_output, _ = self.decoder(encoded_segments, hidden)
        # print("output_shape: ", decoder_output.shape)
        decoder_output = decoder_output.mean(dim=1)  # 각 배치의 모든 세그먼트의 평균을 계산
        # print("adjusted_output_shape: ", decoder_output.shape)
        return decoder_output


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # print(batch_y)
        # print(batch_y.shape)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)  

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(test_loader)

품목별_predictions = {}
품목별_scalers = {}

# 품목명 = '감자'

# train_data, scaler = process_data("./dataset/train/train.csv", 
#                             "./dataset/train/meta/TRAIN_산지공판장_2018-2021.csv", 
#                             "./dataset/train/meta/TRAIN_전국도매_2018-2021.csv", 
#                             품목명)
# train_data.to_csv('filtered_data.csv', index=False, encoding='utf-8-sig')

pbar_outer = tqdm(품목_리스트, desc="품목 처리 중", position=0)
for 품목명 in pbar_outer:
    pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {품목명}")
    train_data, scaler = process_data("/home/work/PFVBG/woo/open/train/train.csv", 
                              "/home/work/PFVBG/woo/open/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                              "/home/work/PFVBG/woo/open/train/meta/TRAIN_전국도매_2018-2021.csv", 
                              품목명)
    품목별_scalers[품목명] = scaler
    
    dataset = AgriculturePriceDataset(train_data)

    # 데이터를 train과 validation으로 분할
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, CFG.batch_size, shuffle=False)
    # print(train_loader)
    input_size = len(dataset.numeric_columns)
    # print(input_size)
    # model = PricePredictionLSTM(input_size, CFG.hidden_size, CFG.num_layers, CFG.output_size)
    
    for batch_x, batch_y in train_loader:
        # print(batch_x.shape)
        _, _, d_model = batch_x.shape
    
    # 모델 초기화
    input_size = 1    # 시계열 데이터의 차원
    hidden_size = 64  # GRU hidden size
    num_layers = 12    # GRU 레이어 수
    output_size = 3   # 출력 차원
    segment_size = 9  # segment 크기
    dropout_p = 0.1   # Dropout 확률

    # # print(batch_x.shape)
    # model = Att().to(device)
    # if 품목명 == '건고추':
    #     model = Att().to(device)
    # else:
    #     model = Att(input_dim=3).to(device)
    # print(model)s

    #품목별 input_dim 정의
    input_dims = {
        '감자': 12,
        '건고추': 1,
        '깐마늘(국산)': 4,
        '대파': 32,
        '무': 2,
        '배추': 1,
        '사과': 4,
        '상추': 29,
        '양파': 12,
        '배': 9
    }

    
    model = Seq2Seq(input_size, hidden_size, output_size, num_layers, segment_size).to(device)
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), CFG.learning_rate)
    
    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)

    train_losses = []
    val_losses = []

    # 품목에 따라 epoch 수 설정
    # if 품목명 == '건고추':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '사과':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '감자':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '배':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '깐마늘(국산)':
    #     epoch_num = 200
    # elif 품목명 == '무':
    #     epoch_num = 200
    # elif 품목명 == '상추':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '배추':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '양파':
    #     epoch_num = CFG.epoch
    # elif 품목명 == '대파':
    #     epoch_num = CFG.epoch
    # else:
    epoch_num = CFG.epoch

    for epoch in range(epoch_num):
        train_loss = train_model(model, train_loader, criterion, optimizer, CFG.epoch)
        val_loss = evaluate_model(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model_{품목명}.pth')
        
        print(f'Epoch {epoch+1}/{epoch_num}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 품목명별 최적 검증 손실 출력 및 파일로 저장
    best_val_loss_message = f'Best Validation Loss for {품목명}: {best_val_loss:.4f}'
    print(best_val_loss_message)
    
    # 결과를 텍스트 파일에 중첩 저장
    # with open(log_file, 'a') as f:
    #     f.write(best_val_loss_message + '\n')
    # 학습 및 검증 손실 시각화
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Over Epochs for {품목명}')
    plt.legend()

    plt.ylim(0, 0.3)

    # 그래프 저장
    plt.savefig(f'./visualization/loss_curve_{품목명}.png')
    plt.show()

    품목_predictions = []

    ### 추론
    model.load_state_dict(torch.load(f'models/best_model_{품목명}.pth', map_location=device))
    pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
    for i in pbar_inner:
        test_file = f"/home/work/PFVBG/woo/open/test/TEST_{i:02d}.csv"
        산지공판장_file = f"/home/work/PFVBG/woo/open/test/meta/TEST_산지공판장_{i:02d}.csv"
        전국도매_file = f"/home/work/PFVBG/woo/open/test/meta/TEST_전국도매_{i:02d}.csv"
        
        test_data, _ = process_data(test_file, 산지공판장_file, 전국도매_file, 품목명, scaler=품목별_scalers[품목명])
        test_dataset = AgriculturePriceDataset(test_data, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                predictions.append(output.cpu().numpy())
        
        predictions_array = np.concatenate(predictions)

        # 예측값을 원래 스케일로 복원
        price_column_index = test_data.columns.get_loc(test_dataset.price_column)
        predictions_reshaped = predictions_array.reshape(-1, 1)
        
        # 가격 열에 대해서만 inverse_transform 적용
        # price_scaler = MinMaxScaler()
        # price_scaler.min_ = 품목별_scalers[품목명].min_[price_column_index]
        # price_scaler.scale_ = 품목별_scalers[품목명].scale_[price_column_index]
        # predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
        #print(predictions_original_scale)

        price_scaler = MinMaxScaler()
        if 품목명 == '건고추':
            price_scaler.min_ = 품목별_scalers[품목명].min_
            price_scaler.scale_ = 품목별_scalers[품목명].scale_
        elif 품목명 == '배추':
            price_scaler.min_ = 품목별_scalers[품목명].min_
            price_scaler.scale_ = 품목별_scalers[품목명].scale_
        else:
            price_scaler.min_ = 품목별_scalers[품목명].min_[price_column_index]
            price_scaler.scale_ = 품목별_scalers[품목명].scale_[price_column_index]
        predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
        
        if np.isnan(predictions_original_scale).any():
            pbar_inner.set_postfix({"상태": "NaN"})
        else:
            pbar_inner.set_postfix({"상태": "정상"})
            품목_predictions.extend(predictions_original_scale.flatten())

            
    품목별_predictions[품목명] = 품목_predictions
    pbar_outer.update(1)


sample_submission = pd.read_csv('/home/work/PFVBG/woo/open/sample_submission.csv')

for 품목명, predictions in 품목별_predictions.items():
    sample_submission[품목명] = predictions

# 결과 저장
sample_submission.to_csv('./dain_128h_9l_9s_.csv', index=False, encoding='utf-8')
