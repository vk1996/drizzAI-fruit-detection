import cv2
from glob import glob
import numpy as np
import random
from matplotlib import pyplot as plt


class Datagen:

    def __init__(self):

        self.class_names=['apple','banana','grapes','orange','strawberry']
        self.scale_factors = [i * 0.1 for i in range(1,6)]
        self.num_max_obj_list=[3,3,3,4,4,5,6,7] #list of max objects of interest that can be in a background image
        #self.num_max_obj_list = [3]  # list of max objects of interest that can be in a background image
        print(self.scale_factors)
        self.bg_files = sorted(glob('/home/vk/personal/drezz-ai-fruits/fruits/bg/*')) #background dir
        self.img_files = sorted(glob('/home/vk/personal/drezz-ai-fruits/fruits/fruits/*/*')) # fruits dir with subfolders like apple , orange
        self.current_coord_intersect =None



    def isIntersect(self,box1,box2):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        # if there is no overlap between predicted and ground-truth box
        # if xB < xA or yB < yA:
        #     return False
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        boxBArea = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        print('iou:',iou)
        # return the intersection over union value
        if iou>0:
            return True
        else:
            return False



    def add_object(self):

        self.img_file = random.choice(self.img_files)
        print('img file:',self.img_file)
        img = cv2.imread(self.img_file)
        scale_factor = random.choice(self.scale_factors)
        img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))

        bg_h = max(int(np.random.uniform(low=1, high=self.bg.shape[0] - img.shape[0] - 10)), 0)
        bg_w = max(int(np.random.uniform(low=1, high=self.bg.shape[1] - img.shape[1] - 10)), 0)
        img_h, img_w = img.shape[0], img.shape[1]
        current_coord=(bg_w,bg_h,bg_w + img_w,bg_h + img_h)
        # current_coord = (bg_h, bg_w, bg_h + img_h,bg_w + img_w)
        if len(self.coords)>0:
            self.current_coord_intersect=[self.isIntersect(coord,current_coord) for coord in self.coords]
            # print(self.current_coord_intersect)
            if any(self.current_coord_intersect):
                # print("Ignoring this coord")
                return

        self.coords.append(current_coord)
        self.classes.append(self.class_names.index(self.img_file.split('/')[-2]))
        self.bg[bg_h:bg_h + img_h, bg_w:bg_w + img_w] = img


    def reset_status(self):
        self.coords = []
        self.classes=[]
        self.bg=None
        self.current_coord_intersect = None

    def generate(self,mode,num_images):

        for i in range(num_images):


            bg_file = random.choice(self.bg_files)
            bg_name=bg_file.split('/')[-1][:-4]
            self.bg = cv2.imread(bg_file)
            num_objects=random.choice(self.num_max_obj_list)
            for i in range(num_objects):
                self.add_object()
            img_name = self.img_file.split('/')[-1][:-4]
            output_name=f"/home/vk/personal/drezz-ai-fruits/fruits/datasets/{mode}/{bg_name}_{img_name}_{str(i)}_{str(len(self.coords))}.jpg"
            # print(output_name)
            # print('coords status:',self.current_coord_intersect)
            cv2.imwrite(output_name, np.uint8(self.bg))

            for class_id,coord in zip(self.classes,self.coords):
                #coords (xmin,ymin,xmax,ymax)
                cx= ((coord[0]+coord[2])/2)/self.bg.shape[1]
                cy= ((coord[1]+coord[3])/2)/self.bg.shape[0]
                w=  (coord[2]-coord[0])/self.bg.shape[1]
                h=  (coord[3]-coord[1])/self.bg.shape[0]

                with open(output_name.replace(".jpg",".txt"),'a') as f:
                    f.write(f"{class_id} {cx} {cy} {w} {h}\n")
            self.reset_status()




datagen=Datagen()
datagen.reset_status()
datagen.generate("train",10000)
datagen.reset_status()
datagen.generate("valid",500)