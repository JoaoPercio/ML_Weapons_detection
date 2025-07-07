from ultralytics import YOLO
import os

def main():
    model = YOLO("yolo12s.pt")

    model.train(
    data=yaml_path,
    epochs=120,
    imgsz=640,
    batch=16,
    project="runs",
    name="treinamento-v1",
    device=0,
   
    hsv_h=0.015,       
    hsv_s=0.7,         
    hsv_v=0.4,         
    degrees=5.0,       
    translate=0.1,     
    scale=0.5,        
    fliplr=0.5,        
    mosaic=1.0,    
    mixup=0.2,     
)
    
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    #dataset_dir = os.path.abspath("")

    dataset_yaml = f"""path: {dataset_dir}
train: images/train
val: images/test

nc: 4
names: ['pistol', 'rifle', 'knife', 'shotgun']
"""

    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(dataset_yaml)

    main()