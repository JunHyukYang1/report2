import cv2
import time
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
import paho.mqtt.client as mqtt
import json

# 모델 및 레이블 파일 경로
model_path = 'test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
label_path = 'test_data/coco_labels.txt'

# 레이블 파일 로드
def load_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if lines[0].split()[0].isdigit():
            return {int(line.split()[0]): line.strip().split(maxsplit=1)[1] for line in lines}
        else:
            return {i: line.strip() for i, line in enumerate(lines)}

labels = load_labels(label_path)

# Edge TPU 인터프리터 로드
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# 카메라 초기화
cap = cv2.VideoCapture(0)

# MQTT 설정
broker = "192.168.0.11"
port = 1883
topic_detection = "detection/info"

client = mqtt.Client()
client.connect(broker, port)
client.loop_start()

last_sent_time = time.time()

try:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 전처리
        input_size = common.input_size(interpreter)
        resized_frame = cv2.resize(frame, input_size)
        common.set_input(interpreter, resized_frame)

        # 객체 검출 수행
        interpreter.invoke()
        objs = detect.get_objects(interpreter, 0.4)  # 신뢰도 임계값을 0.4로 설정

        detection_info = {
            "person_detected": False,
            "objects_detected": False,
            "objects": []
        }

        if objs:
            detection_info["objects_detected"] = True
            for obj in objs:
                label = labels.get(obj.id, obj.id)
                detection_info["objects"].append(label)
                if label == "person":
                    detection_info["person_detected"] = True

        # 5초 간격으로 객체 감지 정보를 MQTT로 전송
        current_time = time.time()
        if current_time - last_sent_time >= 5:
            client.publish(topic_detection, json.dumps(detection_info))
            last_sent_time = current_time

        # 결과 프레임을 보여주기 (디버깅용)
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")  # FPS 출력 추가
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Program interrupted.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()
    client.disconnect()
