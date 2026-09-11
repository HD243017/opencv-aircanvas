import cv2
import numpy as np
import mediapipe as mp

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

PINCH_THRESHOLD = 35

hands = mp_hands.Hands(
    max_num_hands=1,                # 감지할 손 개수
    min_detection_confidence=0.7,   # 손 탐지 신뢰도 임계값
    min_tracking_confidence=0.7     # 이전 위치 기반 프레임 별 손 추적 신뢰도 임계값
)

# createTrackbar의 매개변수에 None이나 비어있으면 오류나서 만듬(빈껍데기)
def nothing(x):
    pass

def choose_color(current_color):
    cv2.namedWindow('Color Palette')
    
    # 트랙바 생성 (초기값은 현재 색상)
    cv2.createTrackbar('R', 'Color Palette', current_color[2], 255, nothing)
    cv2.createTrackbar('G', 'Color Palette', current_color[1], 255, nothing)
    cv2.createTrackbar('B', 'Color Palette', current_color[0], 255, nothing)

    while True:
        # 트랙바 값 읽어오기
        r = cv2.getTrackbarPos('R', 'Color Palette')
        g = cv2.getTrackbarPos('G', 'Color Palette')
        b = cv2.getTrackbarPos('B', 'Color Palette')
        
        # 미리보기 이미지 생성
        img = np.zeros((150, 300, 3), np.uint8)
        img[:] = [b, g, r]  # OpenCV는 BGR 순서 사용

        # 안내 문구
        text_color = (0, 0, 0) if (r + g + b) > 384 else (255, 255, 255)
        cv2.putText(img, "Press ENTER to Apply", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        cv2.imshow('Color Palette', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):  # Enter(13) 또는 Space(32) 누르면 적용
            cv2.destroyWindow('Color Palette')
            return (b, g, r)
        elif key == 27:  # ESC 누르면 취소 (원래 색상 유지)
            cv2.destroyWindow('Color Palette')
            return current_color

## 손 인식
def detect_hand(frame, model):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # mediapipe은 RGB 기반
    results = model.process(rgb_frame)                  # 추론 실행

    # 손 감지
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]          #가장 첫 번째 손만 리턴
    return None

def detect_blue_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HSV 영역
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 노이즈 제거
    mask = cv2.erode(mask, None, iterations=2)  # 침식
    mask = cv2.dilate(mask, None, iterations=2) # 팽창

    # 윤곽선 검출 및 크기 필터링
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)  # 가장큰 파랑 찾기
        if cv2.contourArea(c) > 500:            # 너무 작은 점은 무시
            #파란 부분에서 중심 부분 찾기
            M = cv2.moments(c)                  
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
    return None

def draw_indicator(frame, thumb_pt, index_pt, mid_pt, is_pinched, color, radius, is_eraser):
    overlay = frame.copy()
    cv2.circle(overlay, thumb_pt, 6, (0, 0, 255), -1)
    cv2.circle(overlay, index_pt, 6, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    ind_radius = max(radius // 2, 4)
    if is_pinched:
        if is_eraser:
            # 지우개로 지우는 중: 회색 채워진 원
            cv2.circle(frame, mid_pt, ind_radius, (100, 100, 100), -1)
        else:
            # 펜으로 그리는 중: 현재 색상 원
            cv2.circle(frame, mid_pt, ind_radius, color, -1)
    else:
        # 대기 상태: 흰색 빈 원
        cv2.circle(frame, mid_pt, ind_radius, (255, 255, 255), 2)

def draw_line(canvas, prev_pt, curr_pt, color, thickness):
    if prev_pt is not None:
        cv2.line(canvas, prev_pt, curr_pt, color, thickness)

def merge_canvas(frame, canvas): ## 검은색 바탕의 드로잉을 원본 영상에 합성
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # 마스크를 이용해 배경에서 그림 영역을 파내고 선 색을 채워 넣음
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    return cv2.add(frame_bg, canvas_fg)

def draw_ui(img, color, thickness, is_eraser, track_mode):
    # 상단 UI
    mode_text = "[ERASER]" if is_eraser else "[PEN]"
    info_text = f"Mode: {mode_text} (E)  |  Size: {thickness} (I/O)  |  Clear: C  |  Color: P"
    cv2.putText(img, info_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    sample_color = (80, 80, 80) if is_eraser else color
    cv2.circle(img, (520, 25), max(thickness // 2, 4), sample_color, -1)
    cv2.circle(img, (520, 25), max(thickness // 2, 4) + 1, (255, 255, 255), 1)

    # 하단 UI (현재 인식 모드 표시)
    h, w = img.shape[:2]
    mode_display = "HAND (Pinch)" if track_mode == "HAND" else "COLOR (Blue Object)"
    cv2.putText(img, f"Tracking Mode: {mode_display} [W to switch]", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

def main():
    # 1. 기본 웹캠(0번) 연결
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera open failed!")
        return

    canvas = None
    prev_pt = None
    brush_color = (0, 255, 0)
    brush_thickness = 6
    is_eraser = False

    track_mode = "HAND"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        draw_pt = None
        is_drawing = False

        # 손 인식 기반
        if track_mode == "HAND":
            landmarks = detect_hand(frame, hands)
            if landmarks:
                thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_pt = (int(thumb.x * w), int(thumb.y * h))
                index_pt = (int(index.x * w), int(index.y * h))
                mid_pt = ((thumb_pt[0] + index_pt[0]) // 2, (thumb_pt[1] + index_pt[1]) // 2)

                distance = np.hypot(thumb_pt[0] - index_pt[0], thumb_pt[1] - index_pt[1])
                is_drawing = distance < PINCH_THRESHOLD
                draw_pt = mid_pt
                
                # 손가락 인디케이터 그리기
                draw_indicator(frame, thumb_pt, index_pt, mid_pt, is_drawing, brush_color, brush_thickness, is_eraser)
        
        # 색상 인식 기반
        elif track_mode == "COLOR":
            blue_pt = detect_blue_object(frame)
            if blue_pt:
                is_drawing = True
                draw_pt = blue_pt
                
                # 파란 물체 위에 브러시 표시
                ind_radius = max(brush_thickness // 2, 4)
                ind_color = (100, 100, 100) if is_eraser else brush_color
                cv2.circle(frame, blue_pt, ind_radius, ind_color, -1)
                cv2.circle(frame, blue_pt, ind_radius + 2, (255, 255, 255), 2)

        # 공통 로직
        if is_drawing and draw_pt:
            draw_color = (0, 0, 0) if is_eraser else brush_color
            draw_line(canvas, prev_pt, draw_pt, draw_color, brush_thickness)
            prev_pt = draw_pt
        else:
            prev_pt = None

        # 화면 합성 및 UI 출력
        output = merge_canvas(frame, canvas)
        draw_ui(output, brush_color, brush_thickness, is_eraser, track_mode)

        cv2.imshow('Air Canvas', output)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC: 종료
            break
        elif key in (ord('c'), ord('C')):  # 캔버스 초기화
            canvas = np.zeros_like(frame)
        elif key in (ord('e'), ord('E')):  # 지우개 모드 토글
            is_eraser = not is_eraser
        elif key in (ord('i'), ord('I')):  # 두께 증가
            brush_thickness = min(brush_thickness + 2, 30)
        elif key in (ord('o'), ord('O')):  # 두께 감소
            brush_thickness = max(brush_thickness - 2, 2)
        elif key in (ord('p'), ord('P')):  # 색상 팔레트 열기
            brush_color = choose_color(brush_color)
            is_eraser = False
        elif key in (ord('w'), ord('W')):  # W 키로 모드 변경
            track_mode = "COLOR" if track_mode == "HAND" else "HAND"
            prev_pt = None # 모드가 바뀔 때 엉뚱한 선이 그어지는 것 방지

    # 5. 자원 해제
    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()