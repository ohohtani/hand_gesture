import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

# Step 1: CSV 파일 로드
csv_path = 'C:/Users/aaaasssdd/Desktop/open/train.csv'
train_data = pd.read_csv(csv_path)  # CSV 파일을 읽어 DataFrame 형태로 저장

# 데이터가 올바르게 로드되었는지 확인
print(train_data.head())  # 첫 5개 행 출력

# Step 2: 커스텀 데이터셋 클래스 정의
class VideoDataset(Dataset):
    def __init__(self, csv_data, transform=None):
        """
        초기화 함수
        :param csv_data: CSV 파일에서 불러온 DataFrame
        :param transform: 데이터 전처리 및 변환을 위한 함수
        """
        self.csv_data = csv_data  # CSV 데이터 저장
        self.transform = transform  # 데이터 전처리 변환 설정

    def __len__(self):
        """
        데이터셋의 전체 길이 반환
        """
        return len(self.csv_data)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터 반환
        """
        # 비디오 파일 경로와 레이블 가져오기
        video_path = self.csv_data.iloc[idx]['path']  # 'path' 열에서 비디오 파일 경로를 가져옴
        label = int(self.csv_data.iloc[idx]['label'])  # 'label' 열에서 레이블을 가져옴

        # 비디오 로드 및 전처리
        video = self.load_video(video_path)  # 비디오 로드 함수 호출

        # 변환 적용 (예: 크기 조정, 정규화 등)
        if self.transform:
            video = self.transform(video)

        return video, label

    def load_video(self, path):
        """
        비디오 파일을 로드하고 처리하는 함수
        :param path: 비디오 파일의 경로
        :return: 비디오 데이터를 PyTorch Tensor 형태로 반환
        """
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(path)
        frames = []  # 프레임을 저장할 리스트

        while True:
            ret, frame = cap.read()  # 비디오의 다음 프레임을 읽음
            if not ret:  # 더 이상 읽을 프레임이 없으면 반복문 종료
                break

            # BGR에서 RGB로 변환 (OpenCV는 BGR 형식으로 이미지를 읽음)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # PyTorch Tensor 형태로 변환 및 저장
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)
            frames.append(frame_tensor)

        cap.release()  # 비디오 캡처 객체 해제

        # 비디오 전체 프레임을 텐서 형태로 반환 (4D 텐서: [시간, 채널, 높이, 너비])
        video_tensor = torch.stack(frames)  # 각 프레임 텐서를 쌓아서 하나의 텐서로 만듦
        return video_tensor

# Step 3: 데이터 전처리 및 변환 정의
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # 모든 프레임을 112x112 크기로 조정 (모델의 입력 크기와 일치시킴)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 기준 정규화
])

# Step 4: 데이터셋 및 데이터로더 생성
train_dataset = VideoDataset(train_data, transform=transform)  # VideoDataset 인스턴스 생성

train_loader = DataLoader(
    train_dataset,  # 사용할 데이터셋
    batch_size=4,   # 한 번에 처리할 데이터 샘플 수
    shuffle=True,   # 데이터셋을 무작위로 섞어서 배치 생성
    num_workers=2   # 데이터 로딩을 위한 병렬 작업자 수 (CPU 코어 수에 맞게 설정)
)

# 데이터 확인 (첫 번째 배치)
for videos, labels in train_loader:
    print(f"비디오 배치 크기: {videos.size()}")  # 비디오 텐서 크기 출력
    print(f"레이블 배치 크기: {labels.size()}")  # 레이블 텐서 크기 출력
    break  # 첫 번째 배치만 확인


###############################################################################################

import torch
import torchvision.models.video as models

# 사전 학습된 MC3_18 모델 로드
model = models.mc3_18(pretrained=True)  # ImageNet으로 사전 학습된 MC3_18 모델 로드

# 모델의 구조 확인 (선택 사항)
print(model)

# 출력 레이어 수정
num_classes = 5  # 손동작 클래스 수 (0~4까지 5개)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA 사용 가능하면 GPU, 아니면 CPU 사용

# 모델을 GPU로 이동
model = model.to(device)

################################################################################################

import torch

# Step 6: 손실 함수 정의
criterion = torch.nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 손실 함수

# AdamW 옵티마이저 정의
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW 옵티마이저 설정

################################################################################################

from tqdm import tqdm  # 진행률 바를 위해 tqdm 라이브러리 사용

# Step 7: 모델 학습 루프
num_epochs = 10  # 학습할 에포크 수

for epoch in range(num_epochs):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0  # 에포크 동안의 누적 손실
    correct_predictions = 0  # 정확한 예측의 수

    # tqdm을 사용하여 진행률 바 추가
    for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # 데이터를 GPU로 이동 (가능한 경우)
        videos, labels = videos.to(device), labels.to(device)

        # 옵티마이저의 기울기 초기화
        optimizer.zero_grad()

        # 모델에 입력값을 전달하여 예측값 계산 (forward pass)
        outputs = model(videos)

        # 손실 계산
        loss = criterion(outputs, labels)  # criterion -> 이전 단계에서 설정한 손실함수

        # 손실에 따른 기울기 계산 (backward pass)
        loss.backward()

        # 옵티마이저가 기울기를 사용하여 가중치 업데이트
        optimizer.step()

        # 현재 배치의 손실을 누적
        running_loss += loss.item() * videos.size(0)

        # 정확도 계산
        _, preds = torch.max(outputs, 1)  # 예측 결과의 최대값(클래스) 찾기
        correct_predictions += torch.sum(preds == labels.data)  # 올바른 예측의 수 누적

    # 에포크별 평균 손실 및 정확도 계산
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_predictions.double() / len(train_loader.dataset)

    # 에포크의 손실과 정확도 출력
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')


################################################################################################


# Step 8: 모델 평가 및 저장
def evaluate_model(model, dataloader):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0.0
    correct_predictions = 0

    # 평가에서는 기울기 계산이 필요하지 않으므로 비활성화
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)

            # 모델에 입력값 전달하여 예측값 계산
            outputs = model(videos)
            loss = criterion(outputs, labels)  # 손실 계산

            total_loss += loss.item() * videos.size(0)

            # 예측값과 실제값 비교
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

    # 전체 데이터셋에 대한 평균 손실 및 정확도 계산
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_predictions.double() / len(dataloader.dataset)

    return avg_loss, accuracy


# 검증 또는 테스트 데이터로 평가
validation_loss, validation_acc = evaluate_model(model, train_loader)  # train_loader를 사용하여 검증

print(f'Validation Loss: {validation_loss:.4f}, Validation Accuracy: {validation_acc:.4f}')

# 모델 저장
torch.save(model.state_dict(), 'hand_gesture_recognition_model.pth')
print('모델이 저장되었습니다: hand_gesture_recognition_model.pth')



###################################################################################################################

# Step 9: 학습된 모델 로드
def load_model(model_path, num_classes):
    # 사전 학습된 모델 로드
    model = models.mc3_18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    
    # 저장된 가중치 로드
    model.load_state_dict(torch.load(model_path))

    # 모델을 평가 모드로 설정
    model.eval()
    
    return model

# 모델 경로 및 클래스 수 정의
model_path = 'hand_gesture_recognition_model.pth'
num_classes = 5  # 손동작 클래스 수 (0~4까지 5개)

# 모델 로드
model = load_model(model_path, num_classes)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Step 9: 새로운 데이터에 대한 추론 수행
def predict(model, dataloader):
    model.eval()  # 모델을 평가 모드로 설정
    predictions = []

    with torch.no_grad():  # 기울기 계산 비활성화
        for videos, _ in dataloader:
            videos = videos.to(device)  # 데이터를 GPU로 이동

            # 모델에 입력값 전달하여 예측값 계산
            outputs = model(videos)

            # 예측값 중 가장 높은 확률의 클래스를 선택
            _, preds = torch.max(outputs, 1)

            # 예측 결과 저장
            predictions.extend(preds.cpu().numpy())

    return predictions

# 예측 수행
test_predictions = predict(model, train_loader)  # 예시로 train_loader를 사용하였으나, 실제로는 테스트 데이터로 교체

# 결과를 저장할 파일 경로 정의
output_file = 'test_results.csv'

# 테스트 데이터 ID 로드
test_data = pd.read_csv('test_data.csv')  # 테스트 데이터에 대한 경로를 실제 경로로 변경하세요.

# 예측 결과를 DataFrame에 추가
test_data['label'] = test_predictions

# 결과를 CSV 파일로 저장
test_data.to_csv(output_file, index=False)

print(f'예측 결과가 저장되었습니다: {output_file}')
