import os

import tkinter as tk
from PIL import Image, ImageTk

import pandas as pd
import numpy as np
import cv2

from utils import cv2_safe_read, cv2_safe_write


def intersection_ratio(box1:list, box2:list):
    '''
    Find the intersection area of box1 and box2 over box2
    '''
    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box2_area = max(0, box2[2]-box2[0])*max(0, box2[3]-box2[1])
    ratio = intersection_area/box2_area
    return ratio

def get_head_img(frame_folder:str, info_dict:dict, show_height=360, show_width=640):
    '''
    Get an head image with head annotations to show in the Pop_up Window
    '''
    frame = cv2_safe_read('%s/%06d.jpg'%(frame_folder, info_dict['frameID']))
    h, w, _ = frame.shape
    xmin, ymin, xmax, ymax = map(int, [info_dict['xmin']*w, info_dict['ymin']*h, info_dict['xmax']*w, info_dict['ymax']*h])
    xmin = max(0, xmin-10)
    ymin = max(0, ymin-10)
    xmax = min(w, xmax+10)
    ymax = min(h, ymax+10)
    head_img = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    head_img = cv2.resize(head_img, (show_width, show_height))
    return head_img

def run_select_head(raw_df:pd.DataFrame, frame_dir:str, skipped_frames=[], 
                    start_frame = 1, end_frame = -1):
    '''
    
    Args:
    raw_df (pd.DataFrame): the raw dataframe with head annotations 
    frame_dir (str): the path to read in frames
    start_frame (int): the frame to start running head selection
    end_frame (int): the frame to end running head selection (-1 for according to the end of given dataframe)

    Returns:
    final_df (pd.DataFrame): the dataframe with head annotation cleaned up
    '''
    skip_prev_f = 1
    skip_follow_f = 15

    end_frame = raw_df.frameID.max() if end_frame<0 else end_frame
    raw_df = raw_df[['frameID', 'xmin', 'ymin', 'xmax', 'ymax']]
    # Select teacher
    person_name = 'teacher'
    anchor_coord=None

    while(anchor_coord is None):
        max_det = len(raw_df[raw_df.frameID==start_frame])
        for selected_idx in range(max_det):
            anchor_info = raw_df[raw_df.frameID==start_frame].iloc[selected_idx]['frameID':'ymax'].to_dict()
            anchor_head_img = get_head_img(frame_dir, anchor_info)
            window =  PopupWindow(anchor_head_img, 
                                '[Anchor Setup]',
                                person_name)

            decision = window.get_result()
            if decision=='Yes': 
                anchor_coord = list(anchor_info.values())[1:]
                break
            if decision=='Skip':
                start_frame = start_frame+skip_follow_f
                break
            
    if anchor_coord is None:
        print("Something is wrong with the selection, program will terminate.")
        return None
    
    p1_result = cleanup_by_anchor(raw_df, frame_dir, person_name, anchor_coord, 
                                  start_frame, end_frame, 
                                  skipped_frames, skip_prev_f=skip_prev_f, skip_follow_f=skip_follow_f)
    if p1_result is None:
        return None # Terminated By User
    p1_df, _ = p1_result

    p1_heads_boxes = p1_df[['frameID', 'xmin', 'ymin', 'xmax', 'ymax']]
    p2_proposal = pd.merge(raw_df, p1_heads_boxes, how = 'left', indicator = True)# Remove used annotations
    p2_proposal = p2_proposal[p2_proposal['_merge']=='left_only']
    del p2_proposal['_merge']

    # Select kid
    person_name = 'kid'
    anchor_coord = None
    while(anchor_coord is None):
        max_det = len(raw_df[raw_df.frameID==start_frame])
        for selected_idx in range(max_det):
            anchor_info = raw_df[raw_df.frameID==start_frame].iloc[selected_idx]['frameID':'ymax'].to_dict()
            anchor_head_img = get_head_img(frame_dir, anchor_info)
            window =  PopupWindow(anchor_head_img, 
                                '[Anchor Setup]',
                                'kid')

            decision = window.get_result()
            if decision=='Yes': 
                anchor_coord = list(anchor_info.values())[1:]
                break
            if decision=='Skip':
                start_frame = start_frame+skip_follow_f
                break
    if anchor_coord is None:
        print("Something is wrong with the selection, program will terminate.")
        return -1
    
    p2_result = cleanup_by_anchor(p2_proposal, frame_dir, person_name, anchor_coord, 
                                  start_frame, end_frame, 
                                  skipped_frames, skip_prev_f=skip_prev_f, skip_follow_f=skip_follow_f)
    if p2_result is None:
        return None
    
    p2_df, _ = p2_result

    final_df = pd.concat([p1_df, p2_df]).sort_values('frameID').reset_index(drop=True)
    return final_df



def cleanup_by_anchor(raw_df:pd.DataFrame, 
                      frame_dir: str,
                      person_name: str, 
                      anchor_coord: list, 
                      start_frame: int, 
                      end_frame:int, 
                      skipped_frames = [],
                      skip_prev_f = 1,
                      skip_follow_f = 50,
                      overlap_upper =  0.60,
                      overlap_lower = 0.2):
    '''
    Args:
    raw_df (pd.DataFrame): the dataframe with raw head annotations
    frame_dir (str): the path to the frames
    person_name (int): the id denoting the target person, by default: 1-teacher, 0-kid
    anchor_coord (list): the coordinates of the bounding box of the target person at the start frame
    start_frame (int): the start frame to classify person
    end_frame (int): the end frame
    skipped_frames (list): the frames to skip
    skip_prev_f (int): the number of frames to skip before the 'Skip' selection
    skip_follow_f (int): the number of frames to skip after the 'Skip' selection
    overlap_upper (float): the upperbound threshold to track the bounding box
    overlap_lower (float): the lowerbound threshold to track the boudning box

    Returns:
    person_df: the desired head annotations of the target person from the start to the end frame
    '''

    person_df = raw_df.copy()
    person_df['personID'] = None
    person_df['missing'] = False
    grouped_df = person_df.groupby('frameID')

    frame_ls = raw_df.frameID.unique()
    
    select_new_anchor = False
    confusing_anchor = False
    End = False

    f = start_frame
    while (not End) and (f<end_frame):
        if f not in frame_ls:
            # If encounter missing frames in the dataframe, just increment and ignore
            f+=1 
            continue

        # Case 1: a new anchor needs to be set.
        while select_new_anchor and ((not End) and (f<end_frame)):
            if f not in frame_ls:
                # If encounter missing frames in the dataframe, just increment and ignore
                f+=1 
                continue
            bboxes = grouped_df.get_group(f) # annotations of heads at current frame
            for i, entry in bboxes.iterrows():
                bbox = entry['frameID':'ymax'].to_dict()
                anchor_head_img = get_head_img(frame_dir, bbox)
                window = PopupWindow(anchor_head_img, "[Anchor Setup]" , person_name)
                decision = window.get_result()
                if decision=='Yes': 
                    anchor_coord = entry['xmin':'ymax'].to_list()
                    person_df.at[i, 'personID'] = person_name
                    select_new_anchor = False
                    f+=1
                    break
                elif decision=='No':
                    confusing_anchor = entry['xmin':'ymax'].to_list()
                elif decision=='Skip':
                    break
                elif decision =='Terminate and Drop':
                    End = True
                    select_new_anchor = False
                    break
            if select_new_anchor:
            # After seeing all posible annotations at frame f, 
            # no annotation can be set as the anchor
            # A skip automatically happens
                skip_start = max(1, f-skip_prev_f)
                skip_end = min(f+skip_follow_f, end_frame)
                skipped_frames = skipped_frames+list(range(skip_start, skip_end+1))
                f = skip_end
        
        # Case 2: the given anthor can be used
        bboxes = grouped_df.get_group(f) # annotations of heads at current frame
        overlappings = {}
        confusion = {}
        for i, entry in bboxes.iterrows():
            bbox = entry['xmin':'ymax'].to_list()
            intersect = intersection_ratio(bbox, anchor_coord)
            confusing_intersect = intersection_ratio(bbox, confusing_anchor) if  confusing_anchor else 0
            overlappings[i] = intersect  
            confusion[i] = confusing_intersect

        sorted_items = sorted(overlappings.items(), key=lambda x:x[1])
        sorted_idxes = [item[0] for item in sorted_items]
        sorted_overlappings = [item[1] for item in sorted_items]

        # Case 2.1: No detected head is close to the anchor
        # -> the target head is not found
        if sorted_overlappings[-1]<=overlap_lower:
            f+=1
            continue

        # Case 2.2: there are more than 1 head boxes very close to the given anchor
        # -> the target head may be overlapping with other heads
        elif  (len(sorted_overlappings)>1):
            if (sorted_overlappings[-2] >=overlap_lower) and (confusion[sorted_idxes[-2]]<overlap_lower): 
                select_new_anchor = True
                print("Reset Anchor at frame : %d by case 2.2"%f)
                continue

        # Case 2.4: There seems to be a big movement of head
        elif sorted_overlappings[-1]<overlap_upper:
            select_new_anchor=True
            print("Reset Anchor at frame : %d by case 2.4"%f)
            continue

        # Case 2.5: No other cases:
        target_idx = sorted_idxes[-1]
        person_df.at[target_idx, 'personID'] = person_name
        # update the dictionary of bounding box locations
        anchor_coord  = person_df.loc[target_idx, 'xmin':'ymax'].to_list()
        # increment f
        f+=1

    person_df = person_df[person_df.personID==person_name][['frameID', 'xmin', 'ymin', 'xmax', 'ymax', 'personID', 'missing']]
    # Interpolate and Fill NA values
    interpolated = pd.DataFrame()
    interpolated['frameID'] = range(start_frame, end_frame)

    interpolated = pd.merge(interpolated, person_df, on='frameID', how='left')
    interpolated.loc[interpolated.frameID.isin(skipped_frames), 'missing'] = True
    interpolated = interpolated[(interpolated.missing==False)].apply(lambda x: x.interpolate(method='linear'))

    # Clear-up coordinates of missing heads
    interpolated[interpolated.missing==True].xmin = np.nan
    interpolated[interpolated.missing==True].ymin = np.nan
    interpolated[interpolated.missing==True].xmax = np.nan
    interpolated[interpolated.missing==True].ymax = np.nan

    # interpolated = interpolated[~interpolated.frameID.isin(skipped_frames)]

    # Show final confirmation
    head_at_end = get_head_img(frame_dir, interpolated.iloc[-1].to_dict())
    window = PopupWindow(head_at_end, "[Final Check]", person_name)
    decision = window.get_result()

    if decision=='No':
        print("*"*30)
        print("WARNING!")
        print("Something went wrong in the middle! Unwanted head in the final")
        print("*"*30)
    return interpolated, skipped_frames

class PopupWindow:
    def __init__(self, img_array:np.array, stage: str, person_name:str):
        '''
        Show the window with an image and a question.

        Args:
        img_array (np.array): The image to show (please not in cv2 format)
        stage (str): The stage the pop-up window occurs
        person_name (str): The target person to classify

        Returns:
        decision (str): a decision based on the show up image
        '''
        self.result = None

        # Create Window
        self.master = tk.Tk() 
        self.master.geometry("700x440")

        # Load the image
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(img_pil)

        self.master.update_idletasks()
        self.master.geometry(f'+{50}+{50}')

        # Create the widgets
        self.label = tk.Label(self.master, image=self.photo)
        self.label.grid(row=0, column=0, columnspan=8)

        self.stage_label = tk.Label(self.master, text=stage, fg='black', font=('TkDefaultFont', 10, 'bold'), anchor='w')
        self.stage_label.grid(row=1, column=0)

        self.message_label = tk.Label(self.master, text='Is this a head annotation on the', anchor='e')
        self.message_label.grid(row=1, column=1)

        self.person_label = tk.Label(self.master, text=person_name, fg='red', anchor='w')
        self.person_label.grid(row=1, column=2)


        self.yes_button = tk.Button(self.master, text='Yes', command=self.yes)
        self.yes_button.grid(row=2, column=3)
        
        self.no_button = tk.Button(self.master, text='No', command=self.no)
        self.no_button.grid(row=2, column=4)


        self.skip_button = tk.Button(self.master, text='Skip', command=self.skip)
        self.skip_button.grid(row=2, column=5)


        self.terminate_button = tk.Button(self.master, text='Terminate and Drop', command=self.terminate)
        self.terminate_button.grid(row=2, column=6)
        self.master.mainloop()
        
    def yes(self):
        self.result = 'Yes'
        self.master.destroy()
        
    def no(self):
        self.result = 'No'
        self.master.destroy()
        
    def skip(self):
        self.result = 'Skip'
        self.master.destroy()
        
    def terminate(self):
        self.result = 'Terminate and Drop'
        self.master.destroy()
    

    def get_result(self):
        return self.result

if __name__ =='__main__':
    base_frame_dir =  r'D:\ShanghaiASD_project\ShanghaiASD\20230519\frames'
    base_raw_dir = r'D:\ShanghaiASD_project\ShanghaiASD\20230519\raw_annotations'
    base_df_dir = r'D:\ShanghaiASD_project\ShanghaiASD\20230519\head_annotations'
    # base_cvat_dir = r'D:\ShanghaiASD_project\ShanghaiASD\Annotation_Related\CVAT_partition\Annotation_Unassigned'

    base_visualize_dir = r'D:\ShanghaiASD_project\ShanghaiASD\Test_Visualization'

    vid_id = input("Enter Instance ID: ")
    cameras = os.listdir('%s/%s'%(base_frame_dir, vid_id))
    cameras.sort()
    camera_id = input("Enter Camera ID: ")
    # for camera_id in cameras[1:]:
    skipped_frames = []

    print("Processing %s - %s"%(vid_id, camera_id))

    frame_dir = '%s/%s/%s'%(base_frame_dir, vid_id, camera_id)
    output_path = '%s/%s'%(base_df_dir, vid_id)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    df_path = '%s/%s.csv'%(output_path, camera_id)


    raw_file = '%s/annotations/%s/%s.txt'%(base_raw_dir, vid_id, camera_id)
    columns = ['frameID', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    raw_df = pd.read_csv(raw_file, names = columns)
    raw_df = raw_df.sort_values(by=['frameID', 'xmin'])
    raw_df = raw_df.reset_index(drop=True)

    result = run_select_head(raw_df, frame_dir, skipped_frames=skipped_frames, start_frame=1, end_frame=-1)
    if not (result is None):
        result_df = result
        result_df.to_csv(df_path, index=False)




