import cv2
import numpy as np

from . import detect
from . import IO
from . import transform

'''
Generate
This file contains code for generate data files, used when creating blender project.
A temp storage of calculated data and a way to transfer data to the blender script.
FloorplanToBlender3d
Copyright (C) 2019 Daniel Westberg
'''
# TODO: create big window implementation (nicer windows!)
# TODO: write window and door detection and use the generators for them below!

# Paths to save folder
base_path = "Data/"
path = "Data/"

# Super-Resolution 
Path_pb = ["EDSR_x","ESPCN_x","LapSRN_x","FSRCNN_x"]
meth = ["edsr","espcn","lapsrn","fsrcnn"] 

def generate_all_files(imgpath, info, position=None, rotation=None, CubiCasa=False, SR=[2,"lapsrn"]):
    '''
    Generate all data files
    @Param imgpath
    @Param info, boolean if should be printed
    @Param position, vector of float
    @Param rotation, vector of float
    @Return path to generated file, shape
    '''
    global path
    if CubiCasa == True:
        import numpy as np
        if SR!=None:
            pos = np.argmax(np.array(meth)==SR[1])
            make_res = True
            
        #from skimage import transform
        
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import torch
        import torch.nn.functional as F
        import cv2
        from torch.utils.data import DataLoader
        from model import get_model
        from utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
        from utils.plotting import segmentation_plot, polygons_to_image, draw_junction_from_dict,           discrete_cmap
        discrete_cmap()
        from utils.post_prosessing import split_prediction, get_polygons, split_validation
        from mpl_toolkits.axes_grid1 import AxesGrid

        rot = RotateNTurns()
        room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath",
                        "Entry", "Railing", "Storage", "Garage", "Undefined"]
        icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Applience" ,"Toilet", "Sink",
                        "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

        model = get_model('hg_furukawa_original', 51)
        n_classes = 44
        split = [21, 12, 11]
        model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
        model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
        checkpoint = torch.load('model_best_val_loss_var.pkl')

        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        model.cuda()
        img_path = imgpath

        # Create tensor for pytorch
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # correct color channels

        # Image transformation to range (-1,1)
        img = 2 * (img / 255.0) - 1

        # Move from (h,w,3)--->(3,h,w) as model input dimension is defined like this
        img = np.moveaxis(img, -1, 0)

        # Convert to pytorch, enable cuda
        img = torch.tensor([img.astype(np.float32)]).cuda()
        
        # Super-Resolution
        if make_res == True:
            import cv2 as cv
            from PIL import Image
            import time
            st = time.time()
            sr = cv.dnn_superres.DnnSuperResImpl_create()

            path = "model/super-res/"+Path_pb[pos]+str(SR[0])+".pb"

            sr.readModel(path)

            sr.setModel(meth[pos],SR[0])

            img = sr.upsample(np.array((np.moveaxis(img[0].cpu().data.numpy(), 0, -1)/ 2 + 0.5)*255,dtype='uint8'))
            
            SR_img = img
            img = torch.tensor(np.moveaxis([(img/255-0.5)*2],3,1)).cuda().float()
            
        n_rooms = 12
        n_icons = 11

        with torch.no_grad():
            #Check if shape of image is odd or even
            size_check = np.array([img.shape[2],img.shape[3]])%2

            height = img.shape[2] - size_check[0]
            width = img.shape[3] - size_check[1]

            img_size = (height, width)

            rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
            pred_count = len(rotations)
            prediction = torch.zeros([pred_count, n_classes, height, width])
            for i, r in enumerate(rotations):
                forward, back = r
                # We rotate first the image
                rot_image = rot(img, 'tensor', forward)
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


        rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
        rooms_pred = np.argmax(rooms_pred, axis=0)

        icons_pred = F.softmax(prediction[0, 21+12:], 0).cpu().data.numpy()
        icons_pred = np.argmax(icons_pred, axis=0)

        heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
        polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])
        
        if info:
            print(" ----- Generate ", imgpath, " at pos ", position ," rot ",rotation," -----")

        # Get path to save data
        path = IO.create_new_floorplan_path(base_path)

        shape = generate_floor_file(imgpath, info, SR = make_res, SR_img=SR_img)
        new_shape = generate_walls_file(imgpath, info,CubiCasa = True, polygons=polygons, types=types)
        shape = validate_shape(shape, new_shape)
        new_shape = generate_rooms_file(imgpath, info,CubiCasa = True, room_polygons=room_polygons)
        shape = validate_shape(shape, new_shape)

        #verts, height = generate_big_windows_file(imgpath, info)
        #verts, height = generate_small_windows_file(imgpath, info)
        #verts, height = generate_doors_file(imgpath, info)
        
        #new_shape = generate_big_windows_file(imgpath, info)
        #shape = validate_shape(shape, new_shape)
        #new_shape = generate_small_windows_file(imgpath, info)
        #shape = validate_shape(shape, new_shape)
        #new_shape = generate_doors_file(imgpath, info)
        
        #shape = validate_shape(shape, new_shape)


        transform = generate_transform_file(imgpath, info, position, rotation, shape)

        return path, shape

    if info:
        print(" ----- Generate ", imgpath, " at pos ", position ," rot ",rotation," -----")

    # Get path to save data
    path = IO.create_new_floorplan_path(base_path)

    shape = generate_floor_file(imgpath, info)
    new_shape = generate_walls_file(imgpath, info)
    shape = validate_shape(shape, new_shape)
    new_shape = generate_rooms_file(imgpath, info)
    shape = validate_shape(shape, new_shape)

    #verts, height = generate_big_windows_file(imgpath, info)
    #verts, height = generate_small_windows_file(imgpath, info)
    #verts, height = generate_doors_file(imgpath, info)

    transform = generate_transform_file(imgpath, info, position, rotation, shape)

    return path, shape

def validate_shape(old_shape, new_shape):
    '''
    Validate shape, use this to calculate a objects total shape
    @Param old_shape
    @Param new_shape
    @Return total shape
    '''
    shape = [0,0,0]
    shape[0] = max(old_shape[0], new_shape[0])
    shape[1] = max(old_shape[1], new_shape[1])
    shape[2] = max(old_shape[2], new_shape[2])
    return shape

def get_shape(verts, scale):
    '''
    Get shape
    Rescale boxes to specified scale
    @Param verts, input boxes
    @Param scale to use
    @Return rescaled boxes
    '''
    if len(verts) == 0:
        return [0,0,0]

    posList = transform.verts_to_poslist(verts)
    high = [0,0,0]
    low = posList[0]

    for pos in posList:
        if pos[0] > high[0]:
            high[0] = pos[0]
        if pos[1] > high[1]:
            high[1] = pos[1]
        if pos[2] > high[2]:
            high[2] = pos[2]
        if pos[0] < low[0]:
            low[0] = pos[0]
        if pos[1] < low[1]:
            low[1] = pos[1]
        if pos[2] < low[2]:
            low[2] = pos[2]

    return [high[0] - low[0],high[1] - low[1],high[2] - low[2]]

def generate_transform_file(imgpath, info, position, rotation, shape):
    '''
    Generate transform of file
    A transform contains information about an objects position, rotation.
    @Param imgpath
    @Param info, boolean if should be printed
    @Param position, position vector
    @Param rotation, rotation vector
    @Param shape
    @Return transform
    '''
    #create map
    transform = {}
    if position is None:
        transform["position"] = (0,0,0)
    else:
        transform["position"] = position

    if rotation is None:
        transform["rotation"] = (0,0,0)
    else:
        transform["rotation"] = rotation

    if shape is None:
        transform["shape"] = (0,0,0)
    else:
        transform["shape"] = shape

    IO.save_to_file(path+"transform", transform, info)

    return transform

def generate_icons_file(img_path, info, polygons=[], types = []):
    '''
    Generate room data files
    @Param img_path path to image
    @Param info, boolean if should be printed
    @Return shape
    '''
    icon_polygon_numbers=[i for i,j in enumerate(types) if j['type']=='icon']
    boxes=[]
    for i,j in enumerate(polygons):
        if i in icon_polygon_numbers:
            temp=[]
            for k in j:
                temp.append(np.array([k]))
            boxes.append(np.array(temp))
                
    # Height of waLL
    height = 0.999

    # Scale pixel value to 3d pos
    scale = 100

    # create verts (points 3d), points to use in mesh creations
    verts = []

    # create faces for each plane, describe order to create mesh points
    faces = []

    #Create verts
    icon_count = 0
    for box in boxes:
        verts.extend([transform.scale_point_to_vector(box, scale, height)])
        icon_count+= 1

    # create faces
    for icon in verts:
        count = 0
        temp = ()
        for pos in icon:
            temp = temp + (count,)
            count += 1
        faces.append([(temp)])

    if(info):
        print("Number of icons detected : ", icon_count)

    IO.save_to_file(path+"icon_verts", verts, info)
    IO.save_to_file(path+"icon_faces", faces, info)
    
    dw_polygon_numbers=[i for i,j in enumerate(types) if j['type']=='icon' and j['class'] in [1,2]]
    boxes=[]
    for i,j in enumerate(polygons):
        if i in dw_polygon_numbers:
            temp=[]
            for k in j:
                temp.append(np.array([k]))
            boxes.append(np.array(temp))
    # create verts (points 3d), points to use in mesh creations
    verts = []

    # create faces for each plane, describe order to create mesh points
    faces = []

    #Create verts
    dw_count = 0
    for box in boxes:
        verts.extend([transform.scale_point_to_vector(box, scale, height)])
        dw_count+= 1

    # create faces
    for dw in verts:
        count = 0
        temp = ()
        for pos in dw:
            temp = temp + (count,)
            count += 1
        faces.append([(temp)])
    if(info):
        print("Number of doors/windows detected : ", dw_count)

    IO.save_to_file(path+"dw_verts", verts, info)
    IO.save_to_file(path+"dw_faces", faces, info)

    return get_shape(verts, scale)

def generate_rooms_file(img_path, info, CubiCasa=False, room_polygons=[]):
    '''
    Generate room data files
    @Param img_path path to image
    @Param info, boolean if should be printed
    @Return shape
    '''
    if CubiCasa == True:
        
        boxes = [np.array([[[i.exterior.coords.xy[0][j],i.exterior.coords.xy[1][j]]] for j in range(len(i.exterior.coords.xy[0]))]).astype('int32') for i in room_polygons]
        
        # Height of waLL
        height = 0.999

        # Scale pixel value to 3d pos
        scale = 100

        # create verts (points 3d), points to use in mesh creations
        verts = []
        
        # create faces for each plane, describe order to create mesh points
        faces = []
        
        #Create verts
        room_count = 0
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, height)])
            room_count+= 1

        # create faces
        for room in verts:
            count = 0
            temp = ()
            for pos in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        if(info):
            print("Number of rooms detected : ", room_count)

        IO.save_to_file(path+"rooms_verts", verts, info)
        IO.save_to_file(path+"rooms_faces", faces, info)

        return get_shape(verts, scale)

       
        
    # Read floorplan image
    img = cv2.imread(img_path)

    # grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # create verts (points 3d), points to use in mesh creations
    verts = []
    # create faces for each plane, describe order to create mesh points
    faces = []

    # Height of waLL
    height = 0.999

    # Scale pixel value to 3d pos
    scale = 100

    gray = detect.wall_filter(gray)

    gray = ~gray

    rooms, colored_rooms = detect.find_rooms(gray.copy())

    gray_rooms =  cv2.cvtColor(colored_rooms,cv2.COLOR_BGR2GRAY)

    # get box positions for rooms
    boxes, gray_rooms = detect.detectPreciseBoxes(gray_rooms, gray_rooms)

    #Create verts
    room_count = 0
    for box in boxes:
        verts.extend([transform.scale_point_to_vector(box, scale, height)])
        room_count+= 1

    # create faces
    for room in verts:
        count = 0
        temp = ()
        for pos in room:
            temp = temp + (count,)
            count += 1
        faces.append([(temp)])

    if(info):
        print("Number of rooms detected : ", room_count)

    IO.save_to_file(path+"rooms_verts", verts, info)
    IO.save_to_file(path+"rooms_faces", faces, info)

    return get_shape(verts, scale)

def generate_small_windows_file(img_path, info):
    '''
    Generate small windows data file
    @Param img_path, path to image
    @Param info, boolean if should be printed
    @Return shape
    '''
    # Read floorplan image
    img = cv2.imread(img_path)

    # grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # create verts (points 3d), points to use in mesh creations
    verts = []
    # create faces for each plane, describe order to create mesh points
    faces = []

    # Height of waLL
    height = 1

    # Scale pixel value to 3d pos
    scale = 100

    gray = detect.wall_filter(gray)

    gray = ~gray

    rooms, colored_rooms = detect.find_details(gray.copy())

    gray_rooms =  cv2.cvtColor(colored_rooms,cv2.COLOR_BGR2GRAY)

    # get box positions for rooms
    boxes, gray_rooms = detect.detectPreciseBoxes(gray_rooms, gray_rooms)

    windows = []
    #do a split here, objects next to outside ground are windows, rest are doors or extra space

    for box in boxes:
        if(len(box) >= 4):
            x = box[0][0][0]
            x1 = box[2][0][0]
            y = box[0][0][1]
            y1 = box[2][0][1]

            if (abs(x-x1) > abs(y-y1)):
                windows.append([[[x,round((y+y1)/2)]],[[x1,round((y+y1)/2)]]])
            else:
                windows.append([[[round((x+x1)/2),y]],[[round((x+x1)/2),y1]]])

    '''
    Windows
    '''
    #Create verts for window
    v, faces, window_amount1 = transform.create_nx4_verts_and_faces(windows, height=0.25, scale=scale) # create low piece
    v2, faces, window_amount2 = transform.create_nx4_verts_and_faces(windows, height=1, scale=scale, ground= 0.75) # create heigher piece

    verts = v
    verts.extend(v2)
    window_amount = window_amount1 + window_amount2

    if(info):
        print("Windows created : ", window_amount)


    IO.save_to_file(path+"windows_verts", verts, info)
    IO.save_to_file(path+"windows_faces", faces, info)

    return get_shape(verts, scale)

def generate_doors_file(img_path, info):
    '''
    Generate door data file
    @Param img_path
    @Param info, boolean if should be print
    @Return shape
    '''
    # Read floorplan image
    img = cv2.imread(img_path)

    # grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # create verts (points 3d), points to use in mesh creations
    verts = []
    # create faces for each plane, describe order to create mesh points
    faces = []

    # Height of waLL
    height = 1

    # Scale pixel value to 3d pos
    scale = 100

    gray = detect.wall_filter(gray)

    gray = ~gray

    rooms, colored_rooms = detect.find_details(gray.copy())

    gray_rooms =  cv2.cvtColor(colored_rooms,cv2.COLOR_BGR2GRAY)

    # get box positions for rooms
    boxes, gray_rooms = detect.detectPreciseBoxes(gray_rooms, gray_rooms)

    doors = []

    #do a split here, objects next to outside ground are windows, rest are doors or extra space
    for box in boxes:
        if shape_of_door(point):
            #change doors to actual 2 points instead of 4
            for x,y,x1,y1 in box:
                doors.append([round((x+x1)/2),round((y+y1)/2)])

    '''
    Doors
    '''
    #Create verts for door
    verts, faces, door_amount = transform.create_nx4_verts_and_faces(doors, height, scale)

    if(info):
        print("Doors created : ", door_amount)

    IO.save_to_file(path+"doors_verts", verts, info)
    IO.save_to_file(path+"doors_faces", faces, info)

    return get_shape(verts, scale)

def generate_floor_file(img_path, info, SR=False ,SR_img = None):
    '''
    Generate floor data file
    @Param img_path, path to image
    @Param info, boolean if should be printed
    @Return shape
    '''
    # Read floorplan image
    if SR == True:
        if type(SR_img) == type(None):
            print("No SR-Image given for floor search, Using regular image")
            img = cv2.imread(img_path)
        else:      
            img = SR_img
    else:  
        img = cv2.imread(img_path)

    # grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # detect outer Contours (simple floor or roof solution)
    contour, img = detect.detectOuterContours(gray)

    # create verts (points 3d), points to use in mesh creations
    verts = []
    # create faces for each plane, describe order to create mesh points
    faces = []

    # Height of waLL
    height = 1

    # Scale pixel value to 3d pos
    scale = 100

    #Create verts
    verts = transform.scale_point_to_vector(contour, scale, height)

    # create faces
    count = 0
    for box in verts:
        faces.extend([(count)])
        count += 1


    if(info):
        print("Approximated apartment size : ", cv2.contourArea(contour))

    IO.save_to_file(path+"floor_verts", verts, info)
    IO.save_to_file(path+"floor_faces", faces, info)

    return get_shape(verts, scale)

def generate_walls_file(img_path, info, CubiCasa=False, polygons=[],types=[]):
    '''
    Generate wall data file for floorplan
    @Param img_path, path to input file
    @Param info, boolean if data should be printed
    @Return shape
    '''
    # Get Boxes from CubiCasa workflow
    if CubiCasa == True:
        
        from shapely.geometry import MultiPolygon, Polygon
        from shapely.ops import cascaded_union
        
        Walls = cascaded_union([Polygon(j) for i,j in enumerate(polygons) if types[i]['type']=='wall'])
        
        Icons = cascaded_union([Polygon(j) for i,j in enumerate(polygons) if types[i]['type']=='icon' and types[i]['class'] in [1,2]])
        
        
        polygons = Walls.difference(Icons)
        boxes = [np.array([[[i.exterior.coords.xy[0][j],i.exterior.coords.xy[1][j]]] for j in range(len(i.exterior.coords.xy[0]))]).astype('int32') for i in polygons]

        # create verts (points 3d), points to use in mesh creations
        verts = []
        # create faces for each plane, describe order to create mesh points
        faces = []

        # Height of waLL
        wall_height = 1

        # Scale pixel value to 3d pos
        scale = 100

        # Convert boxes to verts and faces
        verts, faces, wall_amount = transform.create_nx4_verts_and_faces(boxes, wall_height, scale)

        if(info):
            print("Walls created : ", wall_amount)

        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(path+"wall_verts", verts, info)
        IO.save_to_file(path+"wall_faces", faces, info)

        # Create top walls verts
        verts = []
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, 0)])

        # create faces
        faces = []
        for room in verts:
            count = 0
            temp = ()
            for _ in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        # One solution to get data to blender is to write and read from file.
        IO.save_to_file(path+"top_wall_verts", verts, info)
        IO.save_to_file(path+"top_wall_faces", faces, info)

        return get_shape(verts, scale)
            
    # Read floorplan image
    img = cv2.imread(img_path)

    # grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # create wall image (filter out small objects from image)
    wall_img = detect.wall_filter(gray)

    # detect walls
    boxes, img = detect.detectPreciseBoxes(wall_img)

    # create verts (points 3d), points to use in mesh creations
    verts = []
    # create faces for each plane, describe order to create mesh points
    faces = []

    # Height of waLL
    wall_height = 1

    # Scale pixel value to 3d pos
    scale = 100

    # Convert boxes to verts and faces
    verts, faces, wall_amount = transform.create_nx4_verts_and_faces(boxes, wall_height, scale)

    if(info):
        print("Walls created : ", wall_amount)

    # One solution to get data to blender is to write and read from file.
    IO.save_to_file(path+"wall_verts", verts, info)
    IO.save_to_file(path+"wall_faces", faces, info)

    # Create top walls verts
    verts = []
    for box in boxes:
        verts.extend([transform.scale_point_to_vector(box, scale, 0)])

    # create faces
    faces = []
    for room in verts:
        count = 0
        temp = ()
        for _ in room:
            temp = temp + (count,)
            count += 1
        faces.append([(temp)])

    # One solution to get data to blender is to write and read from file.
    IO.save_to_file(path+"top_wall_verts", verts, info)
    IO.save_to_file(path+"top_wall_faces", faces, info)

    return get_shape(verts, scale)