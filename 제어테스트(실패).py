# 뭘해도 2번만 나옴

import cv2
import torch
import numpy as np
import pyautogui
from torchvision import transforms
from PIL import Image
import time
import ctypes
import torchvision.models.video as models

# 시스템 볼륨 조절을 위한 함수 (Windows)
def change_system_volume(volume_change):
    # 볼륨 증가(0xAF) 혹은 감소(0xAE) 키를 입력하는 방식으로 볼륨 조절
    for _ in range(abs(volume_change)):
        key = 0xAF if volume_change > 0 else 0xAE
        ctypes.windll.user32.keybd_event(key, 0, 0, 0)
        time.sleep(0.1)
        ctypes.windll.user32.keybd_event(key, 0, 2, 0)

# 저장된 모델을 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사전 학습된 MC3_18 모델 로드
model = models.mc3_18(pretrained=False)  # 모델 구조 정의 (pretrained=False로 설정)

# 모델 수정: 출력 레이어와 드롭아웃 추가
num_classes = 5
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),  # 50% 드롭아웃 추가
    torch.nn.Linear(model.fc.in_features, num_classes)  # 새로운 출력 레이어 정의
)

# 저장된 가중치 불러오기
model_path = 'C:/Users/tlgjs/OneDrive/바탕 화면/영상처리/hand_gesture_recognition_model.pth'
model.load_state_dict(torch.load(model_path))  # OrderedDict로 저장된 가중치 적용

# 모델을 GPU로 이동
model = model.to(device)
model.eval()

# transform 정의
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 최소한의 움직임이 감지되도록 설정 (백그라운드 프레임을 사용하여 움직임 감지)
def is_significant_frame_change(frame1, frame2, threshold=30):
    frame_diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    blur_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh_diff = cv2.threshold(blur_diff, 25, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresh_diff)
    
    return non_zero_count > threshold

def process_webcam_video():
    cap = cv2.VideoCapture(0)  # 0번 웹캠을 사용

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    max_frames = 30  # 모델이 30 프레임 기준으로 훈련되었으므로 30프레임 사용
    frames = []

    ret, prev_frame = cap.read()  # 이전 프레임 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # 움직임이 있는지 확인 (손이 움직였을 때만 처리)
        if is_significant_frame_change(prev_frame, frame):
            prev_frame = frame.copy()

            # 웹캠에서 읽은 프레임을 RGB로 변환하고 크기 조정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            processed_frame = transform(pil_image)

            frames.append(processed_frame)

            # 30 프레임을 쌓으면 모델에 입력
            if len(frames) == max_frames:
                # 텐서로 변환하고 모델에 입력할 준비
                input_video = torch.stack(frames)  # (T, C, H, W)
                input_video = input_video.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)

                # 모델에 예측 수행
                with torch.no_grad():
                    output = model(input_video)
                    _, pred = torch.max(output, 1)

                pred_class = pred.item()
                print(f"Predicted Class: {pred_class}")

                # 클래스에 따른 동작 수행
                if pred_class == 0:
                    print("Action: Increase Volume")
                    change_system_volume(1)  # 볼륨 증가
                elif pred_class == 1:
                    print("Action: Decrease Volume")
                    change_system_volume(-1)  # 볼륨 감소
                elif pred_class == 2:
                    print("Action: Jump Back 10 seconds")
                    pyautogui.press('left')  # 미디어 재생을 10초 뒤로
                elif pred_class == 3:
                    print("Action: Jump Forward 10 seconds")
                    pyautogui.press('right')  # 미디어 재생을 10초 앞으로
                elif pred_class == 4:
                    print("Action: Play/Pause Video")
                    pyautogui.press('space')  # 미디어 재생 일시정지/재생

                # 프레임 리스트 초기화
                frames = []

        # ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
            break

        # 현재 프레임을 화면에 출력
        cv2.imshow('Webcam', frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_webcam_video()
