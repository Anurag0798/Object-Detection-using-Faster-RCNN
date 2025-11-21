import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

model = fasterrcnn_resnet50_fpn(pretrained = True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

coco_instance_catagory_names = ["unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", 
                                "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", 
                                "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", 
                                "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", 
                                "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", 
                                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
                                "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", 
                                "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
                                "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", 
                                "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", 
                                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", 
                                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]

cap = cv2.VideoCapture(1)

while True :
    _, frame = cap.read()
    img_tensor = F.to_tensor(frame).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score > 0.7:
            x1, y1, x2, y2 = map(int, box)
            class_name = coco_instance_catagory_names[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Display frames", frame)
    if 0xFF == ord("q"):
        break

cap.relase()
cv2.destroyAllWindows()