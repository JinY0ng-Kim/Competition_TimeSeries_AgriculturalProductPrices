from tqdm import tqdm
from data import AgriculturePriceDataset, process_data
import torch
from torch.utils.data import Dataset, DataLoader
from net import Att
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Att()
model.load_state_dict(torch.load('./models/best_model_건고추.pth', map_location=device))
model.eval()
품목_predictions = []

품목명 = '건고추'
품목별_predictions = {}
품목별_scalers = {}

_, scaler = process_data("./dataset/train/train.csv", 
                            "./dataset/train/meta/TRAIN_산지공판장_2018-2021.csv", 
                            "./dataset/train/meta/TRAIN_전국도매_2018-2021.csv", 
                            품목명)
품목별_scalers[품목명] = scaler

### 추론 
pbar_inner = tqdm(range(25), desc="테스트 파일 추론 중", position=1, leave=False)
for i in pbar_inner:
    

    test_file = f"./dataset/test/TEST_{i:02d}.csv"
    산지공판장_file = f"./dataset/test/meta/TEST_산지공판장_{i:02d}.csv"
    전국도매_file = f"./dataset/test/meta/TEST_전국도매_{i:02d}.csv"
    
    test_data, _ = process_data(test_file, 산지공판장_file, 전국도매_file, 품목명, scaler=품목별_scalers[품목명])
    test_dataset = AgriculturePriceDataset(test_data, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch)
            predictions.append(output.numpy())
    
    predictions_array = np.concatenate(predictions)

    # 예측값을 원래 스케일로 복원
    price_column_index = test_data.columns.get_loc(test_dataset.price_column)
    predictions_reshaped = predictions_array.reshape(-1, 1)
    
    # 가격 열에 대해서만 inverse_transform 적용
    price_scaler = MinMaxScaler()
    price_scaler.min_ = 품목별_scalers[품목명].min_[price_column_index]
    price_scaler.scale_ = 품목별_scalers[품목명].scale_[price_column_index]
    predictions_original_scale = price_scaler.inverse_transform(predictions_reshaped)
    #print(predictions_original_scale)
    
    if np.isnan(predictions_original_scale).any():
        pbar_inner.set_postfix({"상태": "NaN"})
    else:
        pbar_inner.set_postfix({"상태": "정상"})
        품목_predictions.extend(predictions_original_scale.flatten())

품목별_predictions[품목명] = 품목_predictions

sample_submission = pd.read_csv('./dataset/sample_submission.csv', encoding='utf-8')

for 품목명, predictions in 품목별_predictions.items():
    sample_submission[품목명] = predictions

# 결과 저장
sample_submission.to_csv('./baseline_v2_0.csv', index=False, encoding='utf-8')