from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import asyncio
import torch
import signal
import sys
import atexit
from ultralytics import YOLO  # YOLOv8 모델이 있는 모듈을 임포트
import logging

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 로깅 수준 설정
logging.getLogger('ultralytics').setLevel(logging.WARNING)  # WARNING 이상의 로그만 출력

# YOLOv8 모델 로드 (verbose 옵션 추가)
model = YOLO('yolov8n.pt', verbose=False)  # YOLOv8 모델 가중치 파일을 로드합니다.

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 밀집도 변수
current_density = 0.0

# 종료 이벤트
stop_event = asyncio.Event()

# 밀집도 계산 함수
def calculate_density(detections, frame):
    height, width, _ = frame.shape
    total_person_area = 0
    for detection in detections:
        if detection['name'] == "person" and detection['confidence'] > 0.5:
            x, y, w, h = detection['box']
            total_person_area += w * h
    frame_area = height * width
    density = total_person_area / frame_area
    return density

# 프레임 얻기 및 밀집도 계산
async def update_frame():
    global current_density
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

       # YOLOv8 모델 추론
        results = model(frame)
        detections = []
        for result in results:
            for detection in result.boxes.data.cpu().numpy():
                x_min, y_min, x_max, y_max = detection[:4]
                confidence = detection[4]
                class_id = int(detection[5])
                if model.names[class_id] == "person" and confidence > 0.5:
                    w = int(x_max - x_min)
                    h = int(y_max - y_min)
                    detections.append({
                        'box': [int(x_min), int(y_min), w, h],
                        'confidence': confidence,
                        'name': model.names[class_id]
                    })
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        current_density = calculate_density(detections, frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        await asyncio.sleep(0.1)  # 프레임 간 간격

# 프로그램 종료 시 실행될 함수
def cleanup():
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()

atexit.register(cleanup)

async def shutdown():
    tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    cleanup()
    sys.exit(0)

def signal_handler(sig, frame):
    asyncio.create_task(shutdown())

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 스트리밍 엔드포인트
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(update_frame(), media_type='multipart/x-mixed-replace; boundary=frame')

# 밀집도 엔드포인트
@app.get("/density")
async def get_density():
    return {"density": current_density}

# index.html 서빙
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    
    config = uvicorn.Config("main:app", host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(server.serve())
    except (KeyboardInterrupt, SystemExit):
        cleanup()
        loop.stop()
        loop.close()
