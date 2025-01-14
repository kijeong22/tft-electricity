import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from metric import QuantileRiskLoss, cal_metrics
from preprocessing import (
    data_load, 
    preprocessing, 
    data_split_scaling, 
    CustomDataset,
    )
from tqdm import tqdm
import os
import glob

def delete_local_files(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted file: {file}")
        except OSError as e:
            print(f"Error: {file} : {e.strerror}")

def post_training_cleanup(model_directory, run_name):

    # delete model files (.pth)
    model_pattern = f'*{run_name}*.pth'
    delete_local_files(model_directory, model_pattern)

class Supervisor():
    def __init__(self, config):

        self.config = config
        self.epochs = config['train']['epochs']
        self.learning_rate = config['train']['learning_rate']
        self.batch_size = config['train']['batch_size']
        self.patient = config['train']['patient']
        self.save_path = config['train']['save_path']
        os.makedirs(self.save_path, exist_ok=True)

        self.device = config['model']['device']
                
        self.fitted = False
        self.model = None
        self.optimizer = None
        self.run_name = config['run_name']

        self.encoder_len = config['data']['encoder_len']
        self.decoder_len = config['data']['decoder_len']

        self.load_data()

        
    def init_model(self):

        if self.model is None:
            raise ValueError("model has not been initialized")

        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.model.to(self.device)
        self.criterion = QuantileRiskLoss(self.tau, self.quantiles, self.device)
        print(f"Number of the Model's parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")


    def load_data(self):

        df = data_load()
        df = preprocessing(df)
        train_set, valid_set, test_set = data_split_scaling(df)
        static_categorical_variables = list(train_set.columns[:1])
        static_continuous_variables = list(train_set.columns[1:3])
        future_variables = list(train_set.columns[3:7])
        past_categorical_variables = list(train_set.columns[3:7])
        past_continuous_variables = list(train_set.columns[7:])
        target = [train_set.columns[-1]]
        variable_dict = {
            'static_categorical_variables' : static_categorical_variables,
            'static_continuous_variables' : static_continuous_variables,
            'past_categorical_variables' : past_categorical_variables,
            'past_continuous_variables' : past_continuous_variables,
            'future_variables' : future_variables,
            'target' : target
        }
        temp_train_set = CustomDataset(train_set, self.encoder_len, self.decoder_len, variable_dict)
        temp_valid_set = CustomDataset(valid_set, self.encoder_len, self.decoder_len, variable_dict)
        temp_test_set = CustomDataset(test_set, self.encoder_len, self.decoder_len, variable_dict)

        self.train_loader = DataLoader(temp_train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(temp_valid_set, batch_size=1, shuffle=False)
        self.test_loader = DataLoader(temp_test_set, batch_size=1, shuffle=False)


    def train(self, data_loader):
        
        self.model.train()

        total_loss = []

        for batch in data_loader:

            static_cate_input, static_conti_input, past_cate_input, past_conti_input, future_input, target = batch

            static_cate_input = static_cate_input.to(self.device)
            static_conti_input = static_conti_input.to(self.device)
            past_cate_input = past_cate_input.to(self.device)
            past_conti_input = past_conti_input.to(self.device)
            future_input = future_input.to(self.device)
            target = target.to(self.device)

            pred, _ = self.model(
                static_cate_input,
                static_conti_input,
                past_cate_input,
                past_conti_input,
                future_input
            )

            loss = self.criterion(target, pred)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if target.shape[0] == self.batch_size:
                weighted_loss = loss.detach().cpu().numpy()
            else:
                weighted_loss = loss.detach().cpu().numpy() * self.batch_size / target.shape[0]

            total_loss.append(weighted_loss)

        train_loss = sum(total_loss)/len(total_loss)

        return train_loss


    def eval(self, data_loader):
        
        self.model.eval()

        total_loss = []
        total_mae = []
        total_rmse = []
        total_mape = []
        total_smape = []

        trues = np.zeros(len(data_loader.dataset), self.decoder_len)
        preds = np.zeros(len(data_loader.dataset), self.decoder_len)

        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                static_cate_input, static_conti_input, past_cate_input, past_conti_input, future_input, target = batch

                static_cate_input = static_cate_input.to(self.device)
                static_conti_input = static_conti_input.to(self.device)
                past_cate_input = past_cate_input.to(self.device)
                past_conti_input = past_conti_input.to(self.device)
                future_input = future_input.to(self.device)
                target = target.to(self.device)

                pred, _ = self.model(static_cate_input,
                                static_conti_input,
                                past_cate_input,
                                past_conti_input,
                                future_input)

                loss = self.criterion(target, pred)
                total_loss.append(loss.detach().cpu().numpy())

                target = target.squeeze().detach().cpu().numpy()
                pred = pred.squeeze().detach().cpu().numpy()

                trues[idx,:] = target
                preds[idx,:] = pred
                                
                performance_dict = cal_metrics(target, pred)
                total_mae.append(performance_dict['MAE'])
                total_rmse.append(performance_dict['RMSE'])
                total_mape.append(performance_dict['MAPE'])
                total_smape.append(performance_dict['SMAPE'])

        eval_loss = sum(total_loss)/len(total_loss)
        eval_metrics = dict({
            "MAE" : sum(total_mae)/len(total_mae),
            "RMSE" : sum(total_rmse)/len(total_rmse),
            "MAPE" : sum(total_mape)/len(total_mape),
            "SMAPE" : sum(total_smape)/len(total_smape)
        })

        return eval_loss, eval_metrics, trues, preds
    

    def fit(self):

        # Training
        patient = 0
        best_valid_loss = float('inf')
        self.model_path = f"{self.save_path}{self.run_name}.pth"
        
        print("Traning...")
        for epoch in tqdm(range(self.epochs)):

            train_loss = self.train(self.train_loader)
            valid_loss, valid_metrics, _, _ = self.eval(self.valid_loader)

            print(f"Epoch: {epoch+1:03} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Valid MAE: {valid_metrics['MAE']:.4f} | Valid RMSE: {valid_metrics['RMSE']:.4f} | Valid MAPE: {valid_metrics['MAPE']:.4f} | Valid SMAPE: {valid_metrics['SMAPE']:.4f}")

            if valid_loss < best_valid_loss:
                best_epoch = epoch
                best_valid_loss = valid_loss
                best_valid_metrics = valid_metrics
                patient = 0

                torch.save(self.model.state_dict(), self.model_path)
                print(f'Model Save!')

            else:
                patient += 1

            if patient == self.patient:
                print("Traning Finish! (Early Stop)")
                print(f"Best Epoch: {best_epoch:.4f}")
                print(f"Best Valid Loss: {best_valid_loss:.4f}")
                print(f'Best Valid MAE: {best_valid_metrics["MAE"]:.4f}')
                print(f'Best Valid RMSE: {best_valid_metrics["RMSE"]:.4f}')
                print(f'Best Valid MAPE: {best_valid_metrics["MAPE"]:.4f}')
                print(f'Best Valid SMAPE: {best_valid_metrics["SMAPE"]:.4f}')

                break

        if patient != self.patient:
            print("Training Finish! (Not Early Stop)")
            print(f"Best Valid Loss: {best_valid_loss:.4f}")
            print(f'Best Valid MAE: {best_valid_metrics["MAE"]:.4f}')
            print(f'Best Valid RMSE: {best_valid_metrics["RMSE"]:.4f}')
            print(f'Best Valid MAPE: {best_valid_metrics["MAPE"]:.4f}')
            print(f'Best Valid SMAPE: {best_valid_metrics["SMAPE"]:.4f}')

        # Testing
        print("Testing...")
        best_params = torch.load(self.model_path)
        self.model.load_state_dict(best_params)
        test_loss, test_metrics, test_trues, test_preds  = self.eval(self.test_loader)

        print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_metrics['MAE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f} | Test MAPE: {test_metrics['MAPE']:.4f} | Test SMAPE: {test_metrics['SMAPE']:.4f}")


    # def test_plot(self):

    #             # 건물별로 145개씩 배치를 처리
    #     num_buildings = 10  # 총 10개의 건물
    #     batch_per_building = 145  # 각 건물당 145개 배치

    #     pred2 = []
    #     # 건물 번호별로 그래프를 그리기 위한 반복문
    #     for building_num in range(num_buildings):
    #         predictions = []
    #         targets = []

    #         # 145개의 배치를 사용해 예측값을 계산
    #         for i, batch in enumerate(test_loader):
    #             if i >= (building_num + 1) * batch_per_building:
    #                 break  # 해당 건물에 해당하는 145개 배치까지만 처리
    #             elif i < building_num * batch_per_building:
    #                 continue  # 이전 건물의 배치는 건너뜀

    #             static_cate_input, static_conti_input, past_cate_input, past_conti_input, future_input, target = batch

    #             with torch.no_grad():
    #                 model.eval()
    #                 # 모델 예측 수행
    #                 pred, _ = model(static_cate_input.to(device),
    #                                 static_conti_input.to(device),
    #                                 past_cate_input.to(device),
    #                                 past_conti_input.to(device),
    #                                 future_input.to(device))

    #             pred = pred.cpu().numpy()  # (1, 24, num_quantiles)
    #             target = target.cpu().numpy()  # (1, 24)

    #             if i % batch_per_building == 0:
    #                 predictions.extend(pred[0])  # 첫 번째 배치는 전체 24개의 시점 저장
    #                 targets.extend(target[0])
    #             else:
    #                 # 이후 배치는 첫 번째 시점 이후 23개의 시점만 사용
    #                 predictions.extend(pred[0, 1:])  # 첫 번째 시점을 제외한 23개 시점 추가
    #                 targets.extend(target[0, 1:])

    #         pred2.append(predictions)

    #         # 예측값과 실제값을 numpy 배열로 변환
    #         predictions = np.array(predictions)[:sequence_len-tau, :]  # (168, num_quantiles)로 자르기
    #         targets = np.array(targets)[:sequence_len-tau]  # (168)으로 자르기

    #         # 시각화 준비
    #         timesteps = np.arange(sequence_len-tau)

    #         fig, ax = plt.subplots(figsize=(16, 6))

    #         # 0.5 Quantile 예측을 포인트로 그리기 (168 시점)
    #         ax.plot(timesteps, predictions[:, 1], label='0.5 Quantile (Point Forecast)', color='#f88379', linestyle='--', marker='.', markersize=4)

    #         # Error bar로 0.1과 0.9 Quantile을 그리기 (168 시점)
    #         yerr = [predictions[:, 1] - predictions[:, 0],  # 0.5 quantile - 0.1 quantile
    #                 predictions[:, 2] - predictions[:, 1]]  # 0.9 quantile - 0.5 quantile
    #         ax.errorbar(timesteps, predictions[:, 1], yerr=yerr, fmt='o', color='#f88379', capsize=3, label='0.1 - 0.9 Quantile Range', markersize=3)

    #         # 실제값 (168 시점의 target)
    #         ax.plot(timesteps, targets, label='True Future Values', color='#08C2FF', marker='x', markersize=4)

    #         # 그래프 설정
    #         ax.set_title(f'Quantile Forecasts vs True Future Values (Building Num: {building_num + 1})', fontsize=14)
    #         ax.set_xlabel('Timesteps', fontsize=15)
    #         ax.set_ylabel('Values', fontsize=15)
    #         plt.xticks(np.arange(0, 169, step=24))
    #         ax.legend(loc='upper right')
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.show()