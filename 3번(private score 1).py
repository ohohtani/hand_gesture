import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms              # 데이터 전처리를 위한 다양한 변환 제공
from torch.utils.data import DataLoader         # 데이터를 배치 단위로 로드하기 위해
import torchvision.models.video as models       # 사전 학습된 비디오 모델 제공
from tqdm import tqdm                           
import cv2                                      # 비디오를 프레임 단위로 처리하기 위해
import os                                       
from sklearn.model_selection import train_test_split

# Step 1: 데이터셋 로드 및 처리
csv_path = 'C:/Users/tlgjs/OneDrive/바탕 화면/영상들/train.csv'
train_data = pd.read_csv(csv_path)

class VideoDataset(Dataset):
    def __init__(self, csv_data, transform=None, base_path='C:/Users/tlgjs/OneDrive/바탕 화면/영상들/'):
        self.csv_data = csv_data
        self.transform = transform
        self.base_path = base_path
        self.resize_transform = transforms.Resize((112, 112))   # 비디오 프레임 크기 조정 (이거 안하니까 112 x 160 이렇게 나오던데 원인 파악 필요)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        relative_path = self.csv_data.iloc[idx]['path']         # csv 데이터의 'path'열에서 데이터를 가져옴
        video_path = os.path.join(self.base_path, relative_path.lstrip('./'))  # 베이스 경로 + path에 있는 경로 합쳐서 절대 경로 완성

        # 비디오 로드
        video = self.load_video(video_path)

        if 'label' in self.csv_data.columns:
            label = int(self.csv_data.iloc[idx]['label'])   # label 열이 존재하는 경우 정수형으로 반환
            return video, label
        else:
            return video                                    # 테스트 데이터의 경우 라벨 없음

    def load_video(self, path, max_frames=30):              # 프레임 수를 일관되게 유지하기 위함인데, 30~60 고려 가능.
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = 0

        if not cap.isOpened():
            print(f"Error: Cannot open video file {path}") 
            return torch.empty(0)

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:        # 비디오 마지막 부분 or 최대 프레임 도달
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR(Opencv 기본값) -> RGB
            frame_rgb_resized = self.resize_transform(transforms.ToPILImage()(frame_rgb)) # RGB로 변환된 이미지를 PIL 형식으로 변환(Resize 사용을 위해 필수)
                                                                                          # 위에 설정되어 있는 (112, 112)로 크기 조정 (self.resize)
            frame_tensor = transforms.ToTensor()(frame_rgb_resized).float()               # 크기 조정한 이미지를 텐서형태로 변환

            frames.append(frame_tensor)       # 텐서형태로 변환된 값을 frames 리스트에 추가하고
            frame_count += 1                  # 프레임의 개수를 1 늘림 (현재까지 처리된)

        cap.release()                         # 비디오 캡쳐 객체 해제

        if len(frames) == 0:                  # 비디오 손상 or 경로 미스의 경우 빈 텐서 반환
            print(f"Warning: No frames found in video {path}")
            return torch.empty(0)

        while len(frames) < max_frames:       # 만약 max_frames 수에 도달하지 못한 경우
            frames.append(frames[-1].clone()) # 마지막 프레임을 복제해서 크기를 맞춤

        video_tensor = torch.stack(frames)[:max_frames]   
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor


if __name__ == '__main__':
    # 데이터셋 분할
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 생성
    train_dataset = VideoDataset(train_data, transform=transform, base_path='C:/Users/tlgjs/OneDrive/바탕 화면/영상들/')
    val_dataset = VideoDataset(val_data, transform=transform, base_path='C:/Users/tlgjs/OneDrive/바탕 화면/영상들/')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 사전 학습된 MC3_18 모델 로드
    model = models.mc3_18(pretrained=True)

    # 모델 수정: 출력 레이어와 드롭아웃 추가
    num_classes = 5
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),  # 50% 드롭아웃 추가
        torch.nn.Linear(model.fc.in_features, num_classes)  # 새로운 출력 레이어 정의
    )

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델을 GPU로 이동
    model = model.to(device)

    # Step 6: 손실 함수 정의
    criterion = torch.nn.CrossEntropyLoss()

    # AdamW 옵티마이저 정의
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Step 7: 모델 학습 루프
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        # tqdm을 사용하여 진행률 바 추가
        for videos, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions.double() / len(train_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # 모델 저장
    torch.save(model.state_dict(), 'hand_gesture_recognition_model.pth')
    print('모델이 저장되었습니다: hand_gesture_recognition_model.pth')

    # 예측 수행을 위한 데이터 로드
    test_csv_path = 'C:/Users/tlgjs/OneDrive/바탕 화면/영상들/test.csv'
    test_data = pd.read_csv(test_csv_path)

    # 올바른 base_path 설정
    test_dataset = VideoDataset(test_data, transform=transform, base_path='C:/Users/tlgjs/OneDrive/바탕 화면/영상들/')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 모델 예측
    model.eval()
    predictions = []

    # 예측 수행
    with torch.no_grad():
        for videos in test_loader:  # 테스트 데이터셋에는 레이블이 없으므로 videos만 가져옴
            videos = videos.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # 결과를 저장할 파일 경로 정의
    output_file = 'C:/Users/tlgjs/OneDrive/바탕 화면/영상들/sample_submission.csv'

    # 'id' 열을 유지하면서 예측 결과를 'label' 열에 추가
    submission_data = pd.DataFrame({'id': test_data['id'], 'label': predictions})

    # 결과를 CSV 파일로 저장
    submission_data.to_csv(output_file, index=False)
    print(f'예측 결과가 저장되었습니다: {output_file}')
