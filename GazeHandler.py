
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import os
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
from gaze_models.model_v1 import ModelSpatial, get_head_box_channel


class GazeDataset(Dataset):
    '''
    Dataset Class
    
    self:
    frame_dir: path to read in frames
    head_boxes_df: dataframe containing head_box annotations
    input_resolution: resolution of transformed image

    Returns:
    ready_img: image after transformation
    ready_head: cropped head image after transformation
    head_channel: head location (as required by model 1)
    '''
    def __init__(self, frame_dir:str, head_boxes_df:pd.DataFrame, input_resolution:int):
        self.frame_dir = frame_dir
        self.head_boxes = head_boxes_df
        self.length = len(self.head_boxes)
        self.input_resolution = input_resolution
        self.transform = transforms.Compose([
                    transforms.Resize((input_resolution, input_resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    def __len__(self):
        return self.length

    def __getitem__(self, i:int):
        # prepare head box
        frameID, xmin, ymin, xmax, ymax, personID= self.head_boxes.iloc[i][['frameID', 'xmin', 'ymin', 'xmax', 'ymax', 'personID']]
        
        # prepare image
        img = Image.open(self.frame_dir+'/%06d.jpg'%frameID)
        width, height = img.size
        
        # prepare head and head channels
        xmin, ymin, xmax, ymax = map(int, [xmin*width, ymin*height, xmax*width, ymax*height])
        head = img.crop([xmin, ymin, xmax, ymax])
        
        ready_img = self.transform(img)
        ready_head = self.transform(head)
        head_channel = get_head_box_channel(xmin, ymin, xmax, ymax, width, height, self.input_resolution, coordconv=False).unsqueeze(0)

        return ready_img, ready_head, head_channel, frameID

def obtain_gaze_point(raw_hm:np.array):
    '''
    Obtain 2D gaze_point from a heatmap (the x-y coordinates with highest value in the heatmap)
    self:
    raw_heatmap: heatmap
    
    Returns:
    max_idx: 2D coordinates ((x,y) in scale 0-1)
    '''
    h, w = raw_hm.shape
    y,x = map(float, np.unravel_index(raw_hm.argmax(), raw_hm.shape))
    max_idx = [x/w, y/h]

    return max_idx

def obtain_gaze_pattern(p1_gaze_data:pd.DataFrame, p2_gaze_data:pd.DataFrame,
                        share_thres = 1e-3, ):
    '''
    Obtaining gaze pattern based on 2D gaze points and heax boxes fo two individuals
    '''
    valid_frames = [i for i in p1_gaze_data.frameID.unique() if i in p2_gaze_data.frameID.unique()] #Obtain frames where both individuals are present
    p1_gaze_data.set_index('frameID', inplace=True)
    p2_gaze_data.set_index('frameID', inplace=True)
    p1_gaze_data['pattern'] = None
    p2_gaze_data['pattern'] = None

    for f in valid_frames:
        p1_info = p1_gaze_data.loc[f]
        p2_info = p2_gaze_data.loc[f]
        p1_head_box = p1_info[['xmin', 'ymin', 'xmax', 'ymax']].values
        p1_gaze = p1_info[['gaze_x', 'gaze_y']].values
        p2_head_box = p2_info[['xmin', 'ymin', 'xmax', 'ymax']].values
        p2_gaze = p2_info[['gaze_x', 'gaze_y']].values
        gaze_dist = np.linalg.norm(p2_gaze-p1_gaze)
        p1_inside = np.all(p1_gaze>=p2_head_box[:2]) and np.all(p1_gaze<=p2_head_box[2:])
        p2_inside = np.all(p2_gaze>=p1_head_box[:2]) and np.all(p2_gaze<=p1_head_box[2:])

        if gaze_dist<share_thres:
            p1_gaze_data.loc[f, 'pattern'] = 'Share'
            p2_gaze_data.loc[f, 'pattern'] = 'Share'
        elif p1_inside and p2_inside:
            p1_gaze_data.loc[f, 'pattern']  = 'Mutual'
            p2_gaze_data.loc[f, 'pattern']  = 'Mutual'
        elif p1_inside:
            p1_gaze_data.loc[f, 'pattern']  = 'Single'
            p2_gaze_data.loc[f, 'pattern']  = 'Miss'
        elif p2_inside:
            p1_gaze_data.loc[f, 'pattern']  = 'Miss'
            p2_gaze_data.loc[f, 'pattern']  = 'Single'
        else:
            p1_gaze_data.loc[f, 'pattern']  = 'Void'
            p2_gaze_data.loc[f, 'pattern']  = 'Void'

    p1_gaze_data.reset_index(inplace=True)
    p2_gaze_data.reset_index(inplace=True)

    return p1_gaze_data, p2_gaze_data


class GazeHandler:
    def __init__(self, 
                 model_weights: str,
                 device_choice='cuda:1', 
                 batch_size=128, 
                 prefetch_factor=2,
                 num_workers=8,
                 input_resolution=224,
                 output_resolution=64,
                 save_gaze_heatmaps=True):
        self.device = torch.device(device_choice)
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.save_gaze_heatmaps = save_gaze_heatmaps

        print("Setting Up Gaze Handler Model...", end= '\t'*4)

        self.gaze_model = ModelSpatial()
        self.gaze_model.to(self.device)
        model_dict = self.gaze_model.state_dict()
        pretrained_dict = torch.load(model_weights)['model']
        model_dict.update(pretrained_dict)
        self.gaze_model.load_state_dict(model_dict)
        self.gaze_model.train(False)
        print("DONE")

    def process(self, head_boxes_file:str, frame_dir:str, heatmap_pkl:str, gaze_file:str):
        '''
        Args:
        head_boxes_file:    Path to the csv file of raw detections of head boxes
        frame_dir:          Directory to frames
        heatmap_pkl:        Path to the heatmap dictionary file 
        gaze_file:          Path to the predicted gaze points and patterns file.


        '''
        head_boxes = pd.read_csv(head_boxes_file)
        head_boxes = head_boxes[~head_boxes.missing] #Only process valid head_box annotations
        assert (not head_boxes.isna().values.any())

        person_list = head_boxes.personID.unique()
        ############################################################################################
        # Run GazeFollow for 1 person each time:
        dfs = []
        for personID in person_list:
            print("Setting up Data for personID %s..."%personID, end= '\t'*4)
            person_head_boxes = head_boxes[head_boxes.personID==personID]
            dataset = GazeDataset(frame_dir, person_head_boxes, self.input_resolution)
            dataloader = DataLoader(dataset=dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        prefetch_factor=self.prefetch_factor,
                                        num_workers=self.num_workers)
            print('DONE')
            print("Running GazeFollow for personID %s..."%personID)
            heatmap_collections = {}
            with torch.no_grad():
                for _, (ready_imgs, ready_heads, head_channels, frameIDs) in enumerate(dataloader):
                    ready_imgs = ready_imgs.cuda().to(self.device)
                    ready_heads = ready_heads.cuda().to(self.device)
                    head_channels = head_channels.cuda().to(self.device)
                    hms, _, inouts = self.gaze_model(ready_imgs, head_channels, ready_heads)
                    hms = hms.squeeze(1)
                    for i in range(len(ready_imgs)):
                        frameID = frameIDs[i].item()
                        heatmap_collections[frameID] = hms[i].cpu().detach().numpy()

            if self.save_gaze_heatmaps:
                output = open('%s_%s.pkl'%(heatmap_pkl, personID), 'wb')
                pickle.dump(heatmap_collections, output)
                output.close()

            person_gaze = []
            for frameID in heatmap_collections.keys():
                gaze_x, gaze_y = obtain_gaze_point(heatmap_collections[frameID])
                person_gaze.append([frameID, gaze_x, gaze_y, personID])
            person_gaze_df = pd.DataFrame(person_gaze, columns=['frameID', 'gaze_x', 'gaze_y', 'personID'])
            combined_df = pd.merge(person_head_boxes, person_gaze_df, on=['frameID', 'personID'])
            dfs.append(combined_df)
            print("DONE")

        
        print("Running Gaze Pattern ...", end= '\t'*4)
        p1_final, p2_final = obtain_gaze_pattern(dfs[0], dfs[1])
        full_df = pd.concat([p1_final, p2_final]).sort_values(['frameID', 'personID']).reset_index(drop=True)
        full_df.to_csv(gaze_file, index=False)
        print("DONE")

