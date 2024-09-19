import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import os
from sklearn.model_selection import train_test_split

# CSV 파일 경로를 설정하여 데이터를 로드
csv_path = 'C:/Users/dusil/OneDrive/바탕 화면/open/train.csv' 
# 모델을 학습시키기 위해서는 train용 영상 데이터와 각 영상이 어떤 동작을 의미하는지 나타내는 train.csv(엑셀파일)이 필요함.
train_data = pd.read_csv(csv_path)
# 위에서 불러온 csv파일을 읽어와서 데이터프레임 형태로 변환하여 train_data에 저장한다.
# 데이터프레임은 쉽게 말하면 pandas 라이브러리에서 제공하는 2차원 테이블 형식의 자료구조이다.

# 비디오 데이터셋 클래스를 정의
class VideoDataset(Dataset):
    def __init__(self, csv_data, transform=None, base_path='C:/Users/dusil/OneDrive/바탕 화면/open/'): # 클래스의 생성자
        # CSV 데이터와 변환기(transform)를 저장
        self.csv_data = csv_data # csv파일 경로
        self.transform = transform 
        self.base_path = base_path # 모델 학습을 위해 필요한 모든 데이터(train, test)가 들어있는 파일의 경로
        # 비디오 프레임을 224x224 크기로 리사이징하는 변환을 설정, 모델에 영상을 입력하기 위해서는 영상들의 크기를 모두 동일하게 만들어야함.(224,224)
        self.resize_transform = transforms.Resize((224, 224))

    # 데이터셋의 크기를 반환
    def __len__(self):
        return len(self.csv_data)

    # 각 비디오 샘플을 로드하고 반환하는 함수
    def __getitem__(self, idx):
        # CSV 파일에서 비디오 경로를 가져와 실제 경로로 변환
        relative_path = self.csv_data.iloc[idx]['path']
        video_path = os.path.join(self.base_path, relative_path.lstrip('./'))

        # 비디오를 로드하여 빠른(Fast) 경로와 느린(Slow) 경로로 반환
        video_fast, video_slow = self.load_video(video_path)

        # 레이블이 있는 경우, 비디오와 함께 레이블을 반환
        if 'label' in self.csv_data.columns:
            label = int(self.csv_data.iloc[idx]['label'])
            return video_fast, video_slow, label
        else:
            return video_fast, video_slow

    # 비디오를 로드하고, Fast와 Slow 경로로 나누어 반환하는 함수
    def load_video(self, path, fast_max_frames=32, slow_max_frames=8):
        # 비디오 파일을 열기
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = 0

        # 비디오 파일이 열리지 않는 경우 에러 메시지를 출력하고 빈 텐서를 반환
        if not cap.isOpened():
            print(f"Error: Cannot open video file {path}") 
            return torch.empty(0), torch.empty(0)

        # 비디오의 각 프레임을 읽어 리스트에 저장
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= fast_max_frames:
                break

            # 프레임을 RGB로 변환하고 크기를 224x224로 리사이즈
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb_resized = self.resize_transform(transforms.ToPILImage()(frame_rgb))
            # 프레임을 텐서로 변환하여 리스트에 추가
            frame_tensor = transforms.ToTensor()(frame_rgb_resized).float()

            frames.append(frame_tensor)
            frame_count += 1

        cap.release()

        # 프레임이 없는 경우 경고 메시지를 출력하고 빈 텐서를 반환
        if len(frames) == 0:
            print(f"Warning: No frames found in video {path}")
            return torch.empty(0), torch.empty(0)

        # Fast 경로의 프레임: 처음 32개 프레임을 사용
        fast_frames = frames[:fast_max_frames]
        while len(fast_frames) < fast_max_frames:
            fast_frames.append(fast_frames[-1].clone())  # 부족한 경우 마지막 프레임을 복사해서 채움

        # Fast 경로의 비디오 텐서를 (채널, 시간, 높이, 너비) 형식으로 변환
        fast_video_tensor = torch.stack(fast_frames)[:fast_max_frames]
        fast_video_tensor = fast_video_tensor.permute(1, 0, 2, 3)

        # Slow 경로의 프레임: 전체 프레임 중 일정 간격으로 추출
        slow_frames = frames[::len(frames) // slow_max_frames][:slow_max_frames]
        while len(slow_frames) < slow_max_frames:
            slow_frames.append(slow_frames[-1].clone())  # 부족한 경우 마지막 프레임을 복사해서 채움

        # Slow 경로의 비디오 텐서를 (채널, 시간, 높이, 너비) 형식으로 변환
        slow_video_tensor = torch.stack(slow_frames)[:slow_max_frames]
        slow_video_tensor = slow_video_tensor.permute(1, 0, 2, 3)

        return fast_video_tensor, slow_video_tensor

if __name__ == '__main__':
    # 데이터셋을 80%는 학습, 20%는 검증 데이터로 분할
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=34)

    # 비디오 데이터 전처리를 위한 변환 설정
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 크기를 128x128로 변경
        transforms.RandomHorizontalFlip(),  # 랜덤 수평 뒤집기
        transforms.RandomRotation(10),  # 랜덤 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변화
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])

    # 학습 및 검증 데이터셋 생성
    train_dataset = VideoDataset(train_data, transform=transform, base_path='C:/Users/dusil/OneDrive/바탕 화면/open/')
    val_dataset = VideoDataset(val_data, transform=transform, base_path='C:/Users/dusil/OneDrive/바탕 화면/open/')

    # 데이터로더 설정: 학습용과 검증용
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    # 사전 학습된 SlowFast 모델을 로드
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

    # 모델의 마지막 레이어를 변경하여 출력 차원을 조정 (클래스 수: 5)
    num_classes = 5
    model.blocks[6].proj = torch.nn.Sequential(
        torch.nn.Dropout(0.5),  # 드롭아웃 추가
        torch.nn.Linear(model.blocks[6].proj.in_features, num_classes)  # 마지막 레이어 수정
    )

    # GPU 사용 가능 여부 확인 및 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델을 GPU로 이동
    model = model.to(device)

    # 손실 함수 정의 (CrossEntropyLoss)
    criterion = torch.nn.CrossEntropyLoss()

    # 옵티마이저 정의 (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

    # 학습 횟수 설정
    num_epochs = 10

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 전환
        running_loss = 0.0
        correct_predictions = 0

        # 학습 데이터셋을 반복하면서 배치 단위로 학습
        for video_fast, video_slow, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            video_fast, video_slow, labels = video_fast.to(device), video_slow.to(device), labels.to(device)
            optimizer.zero_grad()  # 옵티마이저의 기울기 초기화

            # 모델에 Slow와 Fast 입력을 전달하여 출력 계산
            outputs = model([video_slow, video_fast])
            loss = criterion(outputs, labels)  # 손실 계산

            loss.backward()  # 역전파로 기울기 계산
            optimizer.step()  # 옵티마이저로 가중치 갱신

            running_loss += loss.item() * video_fast.size(0)  # 손실 누적
            _, preds = torch.max(outputs, 1)  # 예측 결과 계산
            correct_predictions += torch.sum(preds == labels.data)  # 맞춘 예측의 수

        # 에포크별 손실 및 정확도 출력
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_predictions.double() / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    # 학습된 모델을 파일로 저장
    torch.save(model.state_dict(), 'hand_gesture_recognition_model.pth')
    print('모델이 저장되었습니다: hand_gesture_recognition_model.pth')

    # 예측 수행을 위한 테스트 데이터셋 로드
    test_csv_path = 'C:/Users/dusil/OneDrive/바탕 화면/open/test.csv'
    test_data = pd.read_csv(test_csv_path)

    # 테스트 데이터셋 생성
    test_dataset = VideoDataset(test_data, transform=transform, base_path='C:/Users/dusil/OneDrive/바탕 화면/open/')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 모델을 평가 모드로 전환
    model.eval()
    predictions = []

    # 테스트 데이터에 대해 예측 수행
    with torch.no_grad():  # 기울기 계산을 비활성화
        for video_fast, video_slow in test_loader:
            video_fast, video_slow = video_fast.to(device), video_slow.to(device)
            outputs = model([video_slow, video_fast])  # SlowFast 모델에 맞게 두 경로 입력
            _, preds = torch.max(outputs, 1)  # 예측 결과를 가져옴
            predictions.extend(preds.cpu().numpy())  # CPU로 이동하여 예측 결과 저장

    # 예측 결과를 저장할 파일 경로 설정
    output_file = 'C:/Users/dusil/OneDrive/바탕 화면/open/sample_submission.csv'

    # 테스트 데이터의 'id'와 예측 결과를 저장
    submission_data = pd.DataFrame({'id': test_data['id'], 'label': predictions})

    # 결과를 CSV 파일로 저장
    submission_data.to_csv(output_file, index=False)
    print(f'예측 결과가 저장되었습니다: {output_file}')
