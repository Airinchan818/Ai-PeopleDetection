import cv2
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.ops import nms
from model_architecture import ObjectDetectionModel

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.299,0.224,0.225])])

def decode_target(grid_pred, S=None, image_size=384, conf_thresh=0.5, iou_thresh=0.5, wh_relative_to="image"):
    """
    Decodes predictions to boxes/labels/scores. Accepts either:
      - grid_pred: [S, S, A, 5+C]  (single image)
      - grid_pred: [B, S, S, A, 5+C] (batch)
    Returns:
      - list per image: each item is list of dicts { 'bbox':[x1,y1,x2,y2], 'label':int, 'score':float }
    Notes:
      - x,y,w,h are assumed to be logits that must be activated (sigmoid).
      - w,h interpreted according to wh_relative_to:
          "image" -> w_rel * image_size
          "cell"  -> w_rel * (image_size / S)
    """
  
    single_image = False
    if grid_pred.dim() == 4:         
        grid = grid_pred.unsqueeze(0) 
        single_image = True
    elif grid_pred.dim() == 5:      
        grid = grid_pred
    else:
        raise ValueError(f"Unexpected pred dim {grid_pred.dim()}, expected 4 or 5.")

    B = grid.shape[0]

    if S is None:
        S = grid.shape[1]

    C = grid.shape[-1] - 5
    results_batch = []

    for b in range(B):
        preds = grid[b]                    
        dets = []
        cell_size = image_size / S

        for j in range(S):
            for i in range(S):
                for a in range(preds.shape[2]):
                    cell = preds[j, i, a]   
                    if cell.numel() < 5:

                        continue

                    x_cell = torch.sigmoid(cell[0]).item()
                    y_cell = torch.sigmoid(cell[1]).item()
                    w_val  = cell[2].item()
                    h_val  = cell[3].item()
                    obj_p  = torch.sigmoid(cell[4]).item()


                    if C > 0:
                        class_logits = cell[5:5+C]
                        class_probs = F.softmax(class_logits, dim=0).detach().numpy()
                        class_id = int(class_probs.argmax())
                        class_p = float(class_probs[class_id])
                    else:
                        class_id = 0
                        class_p = 1.0

                    score = obj_p * class_p
                    if score < conf_thresh:
                        continue

                    x_abs = (i + x_cell) * cell_size
                    y_abs = (j + y_cell) * cell_size


                    if wh_relative_to == "cell":
                        w_abs = (torch.sigmoid(torch.tensor(w_val)).item()) * cell_size
                        h_abs = (torch.sigmoid(torch.tensor(h_val)).item()) * cell_size
                    else: 

                        w_abs = (torch.sigmoid(torch.tensor(w_val)).item()) * image_size
                        h_abs = (torch.sigmoid(torch.tensor(h_val)).item()) * image_size

                    x1 = x_abs - w_abs / 2.0
                    y1 = y_abs - h_abs / 2.0
                    x2 = x_abs + w_abs / 2.0
                    y2 = y_abs + h_abs / 2.0


                    x1 = max(0.0, min(image_size, x1))
                    y1 = max(0.0, min(image_size, y1))
                    x2 = max(0.0, min(image_size, x2))
                    y2 = max(0.0, min(image_size, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    dets.append({
                        "bbox":[x1, y1, x2, y2],
                        "label": int(class_id),
                        "score": float(score)
                    })


        if len(dets) > 0:
            boxes = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
            scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
            keep = nms(boxes, scores, iou_thresh).detach().tolist()
            dets = [dets[k] for k in keep]

        results_batch.append(dets)
    return results_batch[0] if single_image else results_batch
class ModelRunners :
    def __init__(self,Model) :
        self.__architecture = ObjectDetectionModel(1)
        self.__checkpoint = torch.load(Model)
        self.__architecture.load_state_dict(self.__checkpoint)
        self.__threshold = 0.25
        self.__imageSize = (720,480)
        self.__camera = 0 
    
    def set_threshold(self) :
        print("Note : for low threshold score can make model so sensitive at prediction ")
        print("Note : for hight threshold score can make model so selective at prediction")
        conf_threshold = float(input("set threshold score System (0.1/0.9) : "))

        self.__threshold = conf_threshold
    
    def __image_procces (self,pred_diction) :

        return decode_target(pred_diction,S=16,image_size=384,conf_thresh=self.__threshold) 
    
    def set_imagesize (self,size : tuple) :
        if not isinstance(size,tuple) :
            raise RuntimeError("the size parameters must be default (720,480)")
        self.__imageSize = size 

    def set_webcam (self,cam : int ) :
        if not isinstance(cam,int) :
            raise RuntimeError("WebCam input must be int (0,1)")
        if cam > 1 : 
            raise RuntimeError("Webcam just aivable for 0 internal and 1 external")
        self.__camera = cam 
    
    def main_model (self) :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__architecture.to(device)
        capture = cv2.VideoCapture(self.__camera)
        while True :
            ret,frame = capture.read()
            if not ret  :
                break 
            H,W = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img_rgb,(384,384))
            tensor = transform(img_resize).unsqueeze(0)  

            with torch.no_grad() :
                pred = self.__architecture(tensor)
            
            result = self.__image_procces(pred[0])
            scaleX,scaleY = W / 384 , H / 384 
            for r in result :
                xmin,ymin,xmax,ymax = r["bbox"]
                xmin,xmax = int(xmin * scaleX) , int(xmax * scaleX)
                ymin,ymax = int(ymin * scaleY), int (ymax * scaleY)

                conf = r["score"] 
                cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                cv2.putText(frame, f"{r['label']} {conf:.2f}",
                            (xmin, max(0, ymin - 5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = cv2.resize(frame,(self.__imageSize))
            cv2.imshow("Live show cam",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
        capture.release()
        cv2.destroyAllWindows()
