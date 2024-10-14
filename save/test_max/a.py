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

from sklearn.preprocessing import MaxAbsScaler

config = {
    "learning_rate": 1e-5,
    "epoch": 2000,
    "batch_size": 512,
    "hidden_size": 64,
    "num_layers": 9,
    "output_size": 3
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_file = './logs.txt'

CFG = SimpleNamespace(**config)

품목_리스트 = ['건고추', '사과', '감자', '배', '깐마늘(국산)', '무', '상추', '배추', '양파', '대파']


def process_data(raw_file, 산지공판장_file, 전국도매_file, 품목명, scaler=None):
    raw_data = pd.read_csv(raw_file)
    산지공판장 = pd.read_csv(산지공판장_file)
    전국도매 = pd.read_csv(전국도매_file)

    # 타겟 및 메타데이터 필터 조건 정의
    conditions = {
    '감자': {
        'target': lambda df: (df['품종명'] == '감자 수미') & (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['감자'], '품종명': ['수미'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['감자'], '품종명': ['수미']}
    },
    '건고추': {
        'target': lambda df: (df['품종명'] == '화건') & (df['거래단위'] == '30 kg') & (df['등급'] == '상품'),
        '공판장': None, 
        '도매': None  
    },
    '깐마늘(국산)': {
        'target': lambda df: (df['거래단위'] == '20 kg') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['마늘'], '품종명': ['깐마늘'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['마늘'], '품종명': ['깐마늘']}
    },
    '대파': {
        'target': lambda df: (df['품종명'] == '대파(일반)') & (df['거래단위'] == '1키로단') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['대파'], '품종명': ['대파(일반)'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['대파'], '품종명': ['대파(일반)']}
    },
    '무': {
        'target': lambda df: (df['거래단위'] == '20키로상자') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['무'], '품종명': ['기타무'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['무'], '품종명': ['무']}
    },
    '배추': {
        'target': lambda df: (df['거래단위'] == '10키로망대') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['배추'], '품종명': ['쌈배추'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['배추'], '품종명': ['배추']}
    },
    '사과': {
        'target': lambda df: (df['품종명'].isin(['홍로', '후지'])) & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['사과'], '품종명': ['후지'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['사과'], '품종명': ['후지']}
    },
    '상추': {
        'target': lambda df: (df['품종명'] == '청') & (df['거래단위'] == '100 g') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['상추'], '품종명': ['청상추'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['상추'], '품종명': ['청상추']}
    },
    '양파': {
        'target': lambda df: (df['품종명'] == '양파') & (df['거래단위'] == '1키로') & (df['등급'] == '상'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['양파'], '품종명': ['기타양파'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['양파'], '품종명': ['양파(일반)']}
    },
    '배': {
        'target': lambda df: (df['품종명'] == '신고') & (df['거래단위'] == '10 개') & (df['등급'] == '상품'),
        '공판장': {'공판장명': ['*전국농협공판장'], '품목명': ['배'], '품종명': ['신고'], '등급명': ['상']},
        '도매': {'시장명': ['*전국도매시장'], '품목명': ['배'], '품종명': ['신고']}
    }
    }

    # 타겟 데이터 필터링
    raw_품목 = raw_data[raw_data['품목명'] == 품목명]
    target_mask = conditions[품목명]['target'](raw_품목)
    filtered_data = raw_품목[target_mask]

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


    # 공판장 데이터 처리
    if conditions[품목명]['공판장']:
        filtered_공판장 = 산지공판장
        for key, value in conditions[품목명]['공판장'].items():
            filtered_공판장 = filtered_공판장[filtered_공판장[key].isin(value)]
        
        filtered_공판장 = filtered_공판장.add_prefix('공판장_').rename(columns={'공판장_시점': '시점'})
        filtered_data = filtered_data.merge(filtered_공판장, on='시점', how='left')

    # 도매 데이터 처리
    if conditions[품목명]['도매']:
        filtered_도매 = 전국도매
        for key, value in conditions[품목명]['도매'].items():
            filtered_도매 = filtered_도매[filtered_도매[key].isin(value)]
        
        filtered_도매 = filtered_도매.add_prefix('도매_').rename(columns={'도매_시점': '시점'})
        filtered_data = filtered_data.merge(filtered_도매, on='시점', how='left')

    # 수치형 컬럼 처리
    numeric_columns = ['평균가격(원)']
    filtered_data = filtered_data[['시점'] + list(numeric_columns)]
    filtered_data[numeric_columns] = filtered_data[numeric_columns].fillna(0)
    # print(filtered_data)

    # 정규화 적용
    # if scaler is None:
    #     scaler = MinMaxScaler()
    #     filtered_data[numeric_columns] = scaler.fit_transform(filtered_data[numeric_columns])
    # else:
    #     filtered_data[numeric_columns] = scaler.transform(filtered_data[numeric_columns])

    if scaler is None:
        scaler = MaxAbsScaler()
        filtered_data[numeric_columns] = scaler.fit_transform(filtered_data[numeric_columns])
    else:
        filtered_data[numeric_columns] = scaler.transform(filtered_data[numeric_columns])

    # print(filtered_data)
    return filtered_data, scaler


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
    def __init__(self, d_model = 90, num_heads=9, max_len=5000):
        super(Att, self).__init__()
        self.in_channels = 9
        self.out_channels = 180
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=9, out_channels=self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels*2, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.out_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.out_channels*2, out_channels=self.out_channels*2*2, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.out_channels*2*2),
            nn.ReLU(inplace=True),
        )

        self.embedding = nn.Linear(720, d_model)  # Learnable Embedding (1차원 -> d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)  # 포지셔널 인코딩
        self.attention_x = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.001)
        self.attention_x1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.001)
        self.attention_x2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=0.001)
    

        self.fc = nn.Linear(90, 3)

    def forward(self, x):
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

pbar_outer = tqdm(품목_리스트, desc="품목 처리 중", position=0)
for 품목명 in pbar_outer:
    pbar_outer.set_description(f"품목별 전처리 및 모델 학습 -> {품목명}")
    train_data, scaler = process_data("/home/work/PFVBG/yong/a/dataset/train/train.csv", 
                              "/home/work/PFVBG/yong/a/dataset/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                              "/home/work/PFVBG/yong/a/dataset/train/meta/TRAIN_전국도매_2018-2021.csv", 
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
        _, _, d_model = batch_x.shape
    
    # print(batch_x.shape)
    model = Att().to(device)
    # print(model)s
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), CFG.learning_rate)
    
    best_val_loss = float('inf')
    os.makedirs('models', exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(CFG.epoch):
        train_loss = train_model(model, train_loader, criterion, optimizer, CFG.epoch)
        val_loss = evaluate_model(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'models/best_model_{품목명}.pth')
        
        print(f'Epoch {epoch+1}/{CFG.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # 품목명별 최적 검증 손실 출력 및 파일로 저장
    best_val_loss_message = f'Best Validation Loss for {품목명}: {best_val_loss:.4f}'
    print(best_val_loss_message)
    
    # 결과를 텍스트 파일에 중첩 저장
    with open(log_file, 'a') as f:
        f.write(best_val_loss_message + '\n')
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
        test_file = f"/home/work/PFVBG/yong/a/dataset/test/TEST_{i:02d}.csv"
        산지공판장_file = f"/home/work/PFVBG/yong/a/dataset/test/meta/TEST_산지공판장_{i:02d}.csv"
        전국도매_file = f"/home/work/PFVBG/yong/a/dataset/test/meta/TEST_전국도매_{i:02d}.csv"
        
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
        # price_scaler.min_ = 품목별_scalers[품목명].min_
        # price_scaler.scale_ = 품목별_scalers[품목명].scale_
        # predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
        price_scaler = MaxAbsScaler()
        price_scaler.max_abs_ = 품목별_scalers[품목명].max_abs
        price_scaler.scale_ = 품목별_scalers[품목명].scale_
        predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
        print(predictions_original_scale)
        
        if np.isnan(predictions_original_scale).any():
            pbar_inner.set_postfix({"상태": "NaN"})
        else:
            pbar_inner.set_postfix({"상태": "정상"})
            품목_predictions.extend(predictions_original_scale.flatten())

            
    품목별_predictions[품목명] = 품목_predictions
    pbar_outer.update(1)


sample_submission = pd.read_csv('/home/work/PFVBG/yong/a/dataset/sample_submission.csv')

for 품목명, predictions in 품목별_predictions.items():
    sample_submission[품목명] = predictions

# 결과 저장
sample_submission.to_csv('./baseline_v6__.csv', index=False, encoding='utf-8')
