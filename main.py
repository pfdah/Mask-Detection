import cv2
from PIL import Image
from torch_mtcnn import detect_faces
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F

data_transforms = transforms.Compose([transforms.Resize((150,150)),
                                       transforms.ToTensor(),                                
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),])

def detect(model):
    model.eval()
    vid = cv2.VideoCapture(0)
    while(True):
        try: 
            ret, frame = vid.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)

            bounding_box, faces = detect_faces(im_pil)
            
            bounding_boxes = list(map(int, bounding_box[0]))

            color = (0, 255, 0)

            cv2.rectangle(frame,(bounding_boxes[:2]),(bounding_boxes[2:4]), color, 2)


            x,y = (bounding_boxes[:2])
            w,h = (bounding_boxes[2:4])
            cropped_image = frame[y : h  , x : w ]
            
            pil_image = Image.fromarray(cropped_image)
            l = data_transforms(pil_image)
            l = DataLoader([l])
            dataiter = iter(l)
            l = next(dataiter)
            output = model(l)
            sigmoid  = F.sigmoid(output)
            
            if sigmoid.item() >= 0.5:
                color = (0,255,0)
                cv2.putText(frame, 'Mask', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1,color)
            else:
                color = (0,0,255)
                cv2.putText(frame, 'No mask', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, color)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except :
            continue

    vid.release()
    cv2.destroyAllWindows()

