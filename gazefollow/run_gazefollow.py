

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import os
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import pickle
from model_v1 import ModelSpatial, get_head_box_channel
import argparse

# Setup params
def get_args_parser():
    parser = argparse.ArgumentParser('General Setup', add_help=False)
    # Setup Path
    parser.add_argument('--model_weights', type=str, default = '/home/changfei/gaze_follow/model_gazefollow.pt')
    parser.add_argument('--frame_dir', type=str, default = '/home/changfei/X_Nas/data/ShanghaiASD/20230519_frames_full/98_TD/98瓢虫1号_1')
    parser.add_argument('--output_dir', type=str, default = '/home/changfei/X_Nas/data/ShanghaiASD/20230519_gaze_annotatioans/98_TD/98瓢虫1号_1')
    parser.add_argument('--head_boxes_file', type=str, default ='/home/changfei/X_Nas/data/ShanghaiASD/20230519_head_annotations/98_TD/98瓢虫1号_1.csv')
    # Setup params
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--input_resolution', type=int, default=224)
    parser.add_argument('--output_resolution', type=int, default=64)
    parser.add_argument('--print_every', type=int, default=100)
    # Specify device
    parser.add_argument('--device', type=str, default='cuda:1')

    # (Optional) keep heatmap predictions
    parser.add_argument('--save_gaze_heatmaps', type=bool, default=True)
    parser.add_argument('--specify_person', type=str)

    return parser

class GazeDataset(Dataset):
    '''
    Dataset Class
    
    Args:
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
    Args:
    raw_heatmap: heatmap
    
    Returns:
    max_idx: 2D coordinates ((x,y) in scale 0-1)
    '''
    h, w = raw_hm.shape
    y,x = map(float, np.unravel_index(raw_hm.argmax(), raw_hm.shape))
    max_idx = [x/w, y/h]

    return max_idx

def run_gazefollow(model:nn.Module, device:torch.device, dataloader:DataLoader):
    '''
    Returns:
    heatmap_collections: dictionary with keys = frameIDs, values = heatmap predictions
    '''
    heatmap_collections = {}
    with torch.no_grad():
        for _, (ready_imgs, ready_heads, head_channels, frameIDs) in enumerate(dataloader):
            ready_imgs = ready_imgs.cuda().to(device)
            ready_heads = ready_heads.cuda().to(device)
            head_channels = head_channels.cuda().to(device)
            hms, _, inouts = model(ready_imgs, head_channels, ready_heads)
            hms = hms.squeeze(1)
            for i in range(len(ready_imgs)):
                frameID = frameIDs[i].item()
                heatmap_collections[frameID] = hms[i].cpu().detach().numpy()
    return heatmap_collections

def main(args):

    ############################################################################################
    # Input Check
    head_boxes = pd.read_csv(args.head_boxes_file)
    head_boxes = head_boxes[~head_boxes.missing] #Only process valid head_box annotations
    
    assert (not head_boxes.isna().values.any()) # Confirm no NaN value in the dataframe
    assert os.path.exists(args.model_weights)
    assert (os.path.exists(args.frame_dir) and len(os.listdir(args.frame_dir))>0)
    os.makedirs(args.output_dir, exist_ok=True)
    ############################################################################################
    # (Optional) Check specific person IDs
    person_list = []
    if args.specify_person:
        person_list = [personID.strip() for personID in args.specify_person.split(',')]
        for personID in person_list:
            assert personID in head_boxes.personID.unique() 
    else:
        person_list = head_boxes.personID.unique()
    ############################################################################################
    # Prepare model
    print("Loading Model...", end= '\t'*4)
    device = torch.device(args.device)
    gaze_model = ModelSpatial()
    gaze_model.to(device)
    model_dict = gaze_model.state_dict()
    pretrained_dict = torch.load(args.model_weights)['model']
    model_dict.update(pretrained_dict)
    gaze_model.load_state_dict(model_dict)
    gaze_model.train(False)
    print("DONE")
    ############################################################################################
    # Run GazeFollow for 1 person each time:
    for personID in person_list:
        print("Setting up Data...", end= '\t'*4)
        person_head_boxes = head_boxes[head_boxes.personID==personID]
        dataset = GazeDataset(args.frame_dir, person_head_boxes, args.input_resolution)
        dataloader = DataLoader(dataset=dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    prefetch_factor=args.prefetch_factor,
                                    num_workers=args.num_workers)
        print('DONE')
        print("Running GazeFollow...")
        person_heatmaps = run_gazefollow(gaze_model, device, dataloader)
        if args.save_gaze_heatmaps:
            output = open('%s/heatmaps_%s.pkl'%(args.output_dir, personID), 'wb')
            pickle.dump(person_heatmaps, output)
            output.close()

        person_gaze = []
        for frameID in person_heatmaps.keys():
            gaze_x, gaze_y = obtain_gaze_point(person_heatmaps[frameID])
            person_gaze.append([frameID, gaze_x, gaze_y, personID])
        person_gaze_df = pd.DataFrame(person_gaze, columns=['frameID', 'gaze_x', 'gaze_y', 'personID'])
        person_gaze_df.to_csv('%s/gazepoints_%s.csv'%(args.output_dir, personID), index=False)
    print("DONE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Running Gaze Follow...', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
