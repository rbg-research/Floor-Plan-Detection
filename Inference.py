
number = 100
data_path = '/home/ubuntu/2Dto3D/CubiCasa5k/data/cubicasa5k/'
import pickle
# Add outer folder 
# Adds higher directory to python modules path.
# Import library
from utils.FloorplanToBlenderLib import *

# Other necessary libraries
import numpy as np
from torch.utils.data import DataLoader
from utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns

def randomPixelAcc(data_path,number=100,seed=4):
    import torch
    from utils import metrics as m
    from model import get_model
    import numpy as np
    import PIL

    model = get_model('hg_furukawa_original', 51)
    n_classes = 44
    split = [21, 12, 11]
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    checkpoint = torch.load('model_best_val_loss_var.pkl')

    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.cuda()
    data_iter= getfromCubiCasa(data_path,number,seed)
    text = 'PixelAccs.txt'
    fo = open(text, "w")
    textfile = []
    
    for i in range(len(data_iter)):
        val=next(data_iter)
        split = [21,12, 11]
        tens=m.get_evaluation_tensors(val, model, split)
        textfile.append("Image"+str(i)+": without post process--->"+str(m.pixel_accuracy(tens[0][1], tens[1][1]))+". With post process--->"+str(m.pixel_accuracy(tens[0][1], tens[2][1])))
                        
    for j in textfile:
        fo.write(i)
        fo.write('\n')
    fo.close()
    return textfile
                        
def getfromCubiCasa(data_path,number,seed):
    
    # # Text file for DataLoader creation

    cat=['high_quality_architectural','high_quality','colorful']
    # path to cubicasa dataset
    data_path = data_path

    import re
    import os
    import numpy as np
    import magic
    import pandas as pd
    import random
    
    sizes=[]
    print(data_path)
    for i in cat:
        files=os.listdir(data_path+i)
        for j in files:
            main=os.listdir(data_path+i+'/'+j)
            for k in main:
                if k=='F1_original.png':
                    t = magic.from_file(data_path+i+'/'+j+'/'+k)
                    t = re.search('(\d+) x (\d+)', t).groups()
                    t = [int(t[0]),int(t[1])]                                                                                                   
                    sizes.append([t,data_path+'/'+i+'/'+j+'/'])
    
    sizes=pd.DataFrame(sizes)

    points=np.array([[sizes[0][i][0],sizes[0][i][1],i] for i in range(5000) if sizes[0][i][0]<800 and sizes[0][i][1]<800])

    random.seed(seed)
    s = random.sample(range(len(points)), 100)

    text = 'images.txt'
    fo = open(text, "w")
    textfile = list(sizes[1][points[:,2][s]])
    for i in textfile:
        fo.write(i)
        fo.write('\n')
    fo.close()

    data_folder = ""
    data_file = text
    normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=False)
    data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
    data_iter = iter(data_loader)
    
    return data_iter



# Import library
from utils.FloorplanToBlenderLib import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
import cv2 
from torch.utils.data import DataLoader




# # 100 images SR vs Original 

'''
get_ipython().run_line_magic('cd', '/home/ubuntu/2Dto3D/GIT/ftb_CubiCasa_v3')
# read images
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn.functional as F
import cv2 
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from floortrans.loaders.house import House
from floortrans.models import get_model
from floortrans.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from floortrans.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict, discrete_cmap
discrete_cmap()
from floortrans.post_prosessing import split_prediction, get_polygons, split_validation
from mpl_toolkits.axes_grid1 import AxesGrid
import time

room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath",
                "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
                "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]



Path_pb = ["EDSR_x2.pb","ESPCN_x2.pb","LapSRN_x2.pb","FSRCNN_x2.pb"]
meth = ["edsr","espcn","lapsrn","fsrcnn"] 
checkpoint = [False,False,False,False]
checkcount = [0,0,0,0]
order = [2,3]

for j in order:
    data_folder = '/home/ubuntu/2Dto3D/CubiCasa5k/data/cubicasa5k/'
    data_file = text
    normal_set = FloorplanSVG(data_folder, data_file, format='txt', original_size=True)
    data_loader = DataLoader(normal_set, batch_size=1, num_workers=0)
    data_iter = iter(data_loader)
    reports = []
        
    print(meth[j] +" has begun")
    st_j = time.time()
    
    count = 0
    failures = []
    length = len(data_loader)
 
    if checkpoint[j]==True:
        
        file_name = "reports_"+meth[j]+"_100_Labels.pkl"
        open_file = open(file_name, "rb")
        reports = pickle.load(open_file)
        open_file.close()
        
        length = 100 - checkcount[j]
        for i in range(checkcount[j]):
            val = next(data_iter)
        count = checkcount[j]
        
    for i in range(length):
        val = next(data_iter)
        junctions = val['heatmaps']
        folder = val['folder'][0]
        image = val['image'].cuda()

        st = time.time()

        # Super Resolution
        import cv2 as cv
        from PIL import Image
        import time
        st = time.time()
        sr = cv.dnn_superres.DnnSuperResImpl_create()

        path = "Super-Resolution/"+Path_pb[j]

        sr.readModel(path)

        sr.setModel(meth[j],2)

        result = sr.upsample(np.array((np.moveaxis(image[0].cpu().data.numpy(), 0, -1)/ 2 + 0.5)*255,dtype='uint8'))

        # Resized image

        resized = cv.resize(np.moveaxis(image[0].cpu().data.numpy(), 0, -1)/ 2 + 0.5,dsize=None,fx=2,fy=2)
        count = count + 1
        for sr in [False,True]:
            # CubiCasa
            (resized - 0.5)*2
            rot = RotateNTurns()
            model = get_model('hg_furukawa_original', 51)

            n_classes = 44
            split = [21, 12, 11]
            model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
            model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
            checkpoint = torch.load('model_best_val_loss_var.pkl')

            model.load_state_dict(checkpoint['model_state'])
            model.eval()
            model.cuda()
            print("Model loaded.")
            with torch.no_grad():
                if sr==True:
                    image = torch.tensor(np.moveaxis([(result/255-0.5)*2],3,1)).cuda().float()
                else:
                    image = val['image'].cuda()
                height = image.shape[2]
                width = image.shape[3]
                img_size = (height, width)
                house = House(data_folder + folder + 'model.svg', height, width)
                # Combining them to one numpy tensor
                label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
                heatmaps = house.get_heatmap_dict()

                label_np = label.data.numpy()    
                rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
                pred_count = len(rotations)
                prediction = torch.zeros([pred_count, n_classes, height, width])
                for i, r in enumerate(rotations):
                    forward, back = r
                    # We rotate first the image
                    rot_image = rot(image, 'tensor', forward)
                    pred = model(rot_image)
                    # We rotate prediction back
                    pred = rot(pred, 'tensor', back)
                    # We fix heatmaps
                    pred = rot(pred, 'points', back)
                    # We make sure the size is correct
                    pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
                    # We add the prediction to output
                    prediction[i] = pred[0]

            prediction = torch.mean(prediction, 0, True)
            rooms_label = label_np[0]
            icons_label = label_np[1]

            rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
            rooms_pred = np.argmax(rooms_pred, axis=0)

            icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
            icons_pred = np.argmax(icons_pred, axis=0)

            icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
            icons_pred = np.argmax(icons_pred, axis=0)
            heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
            
            try:
                polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])
                pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)
            except:
                failures.append(count)
                continue

            from sklearn import metrics
            import pandas as pd
            # Flatten the pixels of the predicted and ground truth label images
            y_true = label_np[0].flatten()
            y_pred = pol_room_seg.flatten()

            room_labels = [0,1,2,3,4,5,6,7,8,9,10,11]
            cm_r = pd.DataFrame(metrics.confusion_matrix(y_true, y_pred,labels=room_labels),columns=room_classes)

            # Print the precision and recall, among other metrics
            cr = metrics.classification_report(y_true, y_pred,labels=room_labels,target_names=room_classes,zero_division=1,output_dict=True)
            cr_r = pd.DataFrame(cr)

            from sklearn import metrics
            import pandas as pd

            # Flatten the pixels of the predicted and ground truth label images
            y_true = label_np[1].flatten()
            y_pred = pol_icon_seg.flatten()

            icon_labels = [0,1,2,3,4,5,6,7,8,9,10]
            cm_i = pd.DataFrame(metrics.confusion_matrix(y_true, y_pred,labels=icon_labels),columns=icon_classes)

            # Print the precision and recall, among other metrics
            cr = metrics.classification_report(y_true, y_pred,labels=icon_labels,target_names=icon_classes,zero_division=1,output_dict=True)
            cr_i = pd.DataFrame(cr)
            reports.append([count, sr, label_np[0],label_np[1],pol_room_seg,pol_icon_seg])

            file_name = "reports_"+meth[j]+"_100_Labels.pkl"


            open_file = open(file_name, "wb")
            pickle.dump(reports, open_file)
            open_file.close()
        print("ITER"+str(count)+": TimeElapsed---->",time.time()-st)

    file_name = "reports_"+meth[j]+"_100_Labels.pkl"


    open_file = open(file_name, "wb")
    pickle.dump(reports, open_file)
    open_file.close()

    file_name = "failure_"+meth[j]+"_100_Labels.pkl"


    open_file = open(file_name, "wb")
    pickle.dump(failures, open_file)
    open_file.close()
    print("----------------> TIME FOR 100 images "+meth[j]+" : ",time.time()-st_j)'''
