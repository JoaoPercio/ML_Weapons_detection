from ultralytics import YOLO
import cv2

MODEL_PATH = 'treinamento-v110/weights/best.pt'  

model = YOLO(MODEL_PATH)

cap = None
temp_cap = cv2.VideoCapture(2)
if temp_cap.isOpened():
    cap = temp_cap
    print(f"Usando a câmera {2}")


if cap is None or not cap.isOpened():
    print("Nenhuma câmera disponível.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame.")
        break

    results = model.predict(frame, conf=0.5) 

   
    annotated_frame = results[0].plot()

    
    cv2.imshow('Detecção com YOLOv12', annotated_frame)

   
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()