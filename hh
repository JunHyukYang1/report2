import paho.mqtt.client as mqtt
import pyttsx3
import json
import time
import queue
import threading

last_tts_time = 0
distance1 = float('inf')
distance2 = float('inf')

engine = pyttsx3.init()

tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def tts(text):
    tts_queue.put(text)

def on_message_distance(client, userdata, message):
    global distance1, distance2
    msg_payload = message.payload.decode()
    print(f"Received message: {msg_payload} on topic {message.topic}")

    try:
        data = json.loads(msg_payload)
        distance1 = data["distance1"]
        distance2 = data["distance2"]
    except json.JSONDecodeError as e:
        print(f"Invalid JSON data: {msg_payload}")
        print(f"Error: {e}")
    except KeyError as e:
        print(f"JSON parsing error: {e}")

def on_message_detection(client, userdata, message):
    global last_tts_time, distance1, distance2
    msg_payload = message.payload.decode()
    print(f"Received message: {msg_payload} on topic {message.topic}")

    try:
        data = json.loads(msg_payload)
        person_detected = data.get("person_detected", False)
        objects_detected = data.get("objects_detected", False)
        current_time = time.time()

        if (current_time - last_tts_time) >= 5:
            if distance1 > 10 and distance2 > 10:
                tts_message = "Safe. No object detected."
                tts(tts_message)
                print(tts_message)
            elif distance1 <= 10 or distance2 <= 10:
                if person_detected:
                    tts_message = "Person in 10 centimeters."
                    tts(tts_message)
                    print(tts_message)
                else:
                    tts_message = "Object in 10 centimeters."
                    tts(tts_message)
                    print(tts_message)

            last_tts_time = current_time

    except json.JSONDecodeError as e:
        print(f"Invalid JSON data: {msg_payload}")
        print(f"Error: {e}")
    except KeyError as e:
        print(f"JSON parsing error: {e}")

broker = "192.168.0.11"
port = 1883
topic_distance = "distance/measured"
topic_detection = "detection/info"

client_distance = mqtt.Client()
client_distance.on_message = on_message_distance
client_distance.connect(broker, port)
client_distance.subscribe(topic_distance)
client_distance.loop_start()

client_detection = mqtt.Client()
client_detection.on_message = on_message_detection
client_detection.connect(broker, port)
client_detection.subscribe(topic_detection)
client_detection.loop_start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Subscriber stopped.")
finally:
    client_distance.loop_stop()
    client_detection.loop_stop()
    client_distance.disconnect()
    client_detection.disconnect()
    tts_queue.put(None)
    tts_thread.join()
