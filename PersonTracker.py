import os

import tkinter as tk
from PIL import Image, ImageTk

import pandas as pd
import numpy as np
import cv2

from utils import cv2_safe_read, cv2_safe_write

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

class PersonTracker:
    def __init__(self, 
                skip_prev_f = 1,
                skip_follow_f = 15,
                overlap_upper =  0.60,
                overlap_lower = 0.2):        
        '''
        Track and Identify Person
        Args:
        skip_prev_f: when select skip, the number of frames to skipped before the pop-up frame
        skip_follow_f: when select skip, the number of frames to skipped after the pop-up frame
        overlap_upper: the upperbound of deciding a tracked person.
        overlap_lower: the lowerbound of deciding a tracked person.
        '''

        self.skip_prev_f = skip_prev_f
        self.skip_follow_f = skip_follow_f
        self.overlap_upper = overlap_upper
        self.overlap_lower = overlap_lower


        self.tracked_dfs = {}
        self.frame_dir = None
        self.proposal_df = None

    def release(self):
        self.tracked_dfs = {}
        self.frame_dir = None
        self.proposal_df = None

    def get_person_df(self, personID):
        if not personID in self.tracked_dfs:
            print("[ERROR] The specified person %s is not found, please check."%personID)
            print("Tracked personIDs:")
            print(self.tracked_dfs.keys())

        else:
            return self.tracked_dfs[personID]
    
    def get_full_df(self):
        dfs = list(self.tracked_dfs.values())
        full_df = pd.concat(dfs).sort_values('frameID').reset_index(drop=True)
        return full_df
        
    def load_from_files(self,
                        raw_detection_file: str,
                        frame_dir):
        '''
        Args:
        raw_detection_file: the csv file with raw_annotation_detection
        frame_dir:          the path to read in frames
        '''

        self.frame_dir = frame_dir
        columns = ['frameID', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
        raw_df = pd.read_csv(raw_detection_file, names = columns)
        raw_df = raw_df[['frameID', 'xmin', 'ymin', 'xmax', 'ymax']].sort_values(by=['frameID', 'xmin']).reset_index(drop=True)
        self.proposal_df = raw_df

    def track_person(self, personID, start_frame=1, end_frame = -1):
        '''
        Args:
        personID:       the target person to be tracked
        start_frame:    the frame to start running head selection
        end_frame:      the frame to end running head selection (-1 for according to the end of given dataframe)
        '''

        if self.frame_dir is None:
            print("[ERROR] No information stored. Please call the load_from_files function first!")
            return None
        
        else:
            end_frame = self.proposal_df.frameID.max() if end_frame<0 else end_frame
            tracked_df = self.__track_person(
                self.proposal_df,
                personID,
                start_frame,
                end_frame
            )
            if not tracked_df is None:
                self.tracked_dfs[personID] = tracked_df
                new_proposal = pd.merge(self.proposal_df, tracked_df, how = 'left', indicator = True)# Remove used annotations
                new_proposal = new_proposal[new_proposal['_merge']=='left_only']
                del new_proposal['_merge']
                self.proposal_df = new_proposal

            else:
                print("[ERROR] Tracking Failed.")    

    
    def track_all(self, start_frame=1, end_frame=-1):
        '''
        Args:
        Track all people in the proposal df

        personIDs:          the perople to track in the proprosal df, if not specified, will track all people present in the video, by naming 'p1', 'p2', ...
        start_frame:        the frame to start running head selection
        end_frame:          the frame to end running head selection (-1 for according to the end of given dataframe)

        '''
    # TODO
        return None

    
    def __get_head_img(
            self,
            info_dict:dict, 
            show_height=360, 
            show_width=640):
        '''
        Get an head image with head annotations to show in the Pop_up Window
        '''
        frame = cv2_safe_read('%s/%06d.jpg'%(self.frame_dir, info_dict['frameID']))
        h, w, _ = frame.shape
        xmin, ymin, xmax, ymax = map(int, [info_dict['xmin']*w, info_dict['ymin']*h, info_dict['xmax']*w, info_dict['ymax']*h])
        xmin = max(0, xmin-10)
        ymin = max(0, ymin-10)
        xmax = min(w, xmax+10)
        ymax = min(h, ymax+10)
        head_img = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
        head_img = cv2.resize(head_img, (show_width, show_height))
        
        return head_img
    
    def __intersection_ratio(self, box1:list, box2:list):
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
    
    def __track_person(
            self,
            proposal_df:pd.DataFrame,
            personID: str,
            start_frame:int,
            end_frame:int,
        ):
    
        proposal_df['personID'] = None
        skipped_frames = []

        grouped_df = proposal_df.groupby('frameID')
        frame_ls = proposal_df.frameID.unique()
        
        select_new_anchor = True
        confusing_anchor = False
        Terminated = False

        f = start_frame

        while (not Terminated) and (f<end_frame):
            if f not in frame_ls:
                # If encounter missing frames in the dataframe, just increment and ignore
                f+=1 
                continue
            # Case 1: a new anchor needs to be set.
            while select_new_anchor and ((not Terminated) and (f<end_frame)):
                if f not in frame_ls:
                    # If encounter missing frames in the dataframe, just increment and ignore
                    f+=1 
                    continue
                bboxes = grouped_df.get_group(f) # annotations of heads at current frame
                show_window = True
                for i, entry in bboxes.iterrows():
                    bbox = entry['frameID':'ymax'].to_dict()
                    if show_window:
                        anchor_head_img = self.__get_head_img(bbox)
                        window = PopupWindow(anchor_head_img, "[Anchor Setup]" , personID)
                        decision = window.get_result()
                        if decision=='Yes': 
                            anchor_coord = entry['xmin':'ymax'].to_list()
                            proposal_df.at[i, 'personID'] = personID
                            select_new_anchor = False
                            f+=1
                            show_window = False
                        elif decision=='No':
                            confusing_anchor = entry['xmin':'ymax'].to_list()
                        elif decision=='Skip':
                            show_window = False
                        elif decision =='Terminate and Drop':
                            Terminated = True
                            select_new_anchor = False
                            show_window = False
                    else:
                        confusing_anchor = entry['xmin':'ymax'].to_list()
                    
                if select_new_anchor:
                # After seeing all posible annotations at frame f, 
                # no annotation can be set as the anchor
                # A skip automatically happens
                    skip_start = max(1, f-self.skip_prev_f)
                    skip_end = min(f+self.skip_follow_f, end_frame)
                    skipped_frames = skipped_frames+list(range(skip_start, skip_end+1))
                    f = skip_end
            
            # Case 2: the given anthor can be used
            bboxes = grouped_df.get_group(f) # annotations of heads at current frame
            overlappings = {}
            confusion = {}
            for i, entry in bboxes.iterrows():
                bbox = entry['xmin':'ymax'].to_list()
                intersect = self.__intersection_ratio(bbox, anchor_coord)
                confusing_intersect = self.__intersection_ratio(bbox, confusing_anchor) if  confusing_anchor else 0
                overlappings[i] = intersect  
                confusion[i] = confusing_intersect

            sorted_items = sorted(overlappings.items(), key=lambda x:x[1])
            sorted_idxes = [item[0] for item in sorted_items]
            sorted_overlappings = [item[1] for item in sorted_items]

            # Case 2.1: No detected head is close to the anchor
            # -> the target head is not found
            if sorted_overlappings[-1]<=self.overlap_lower:
                f+=1
                continue

            # Case 2.2: there are more than 1 head boxes very close to the given anchor
            # -> the target head may be overlapping with other heads
            elif  (len(sorted_overlappings)>1):
                if (sorted_overlappings[-2] >=self.overlap_lower) and (confusion[sorted_idxes[-2]]<self.overlap_lower): 
                    select_new_anchor = True
                    print("Reset Anchor at frame : %d by case 2.2"%f)
                    continue

            # Case 2.4: There seems to be a big movement of head
            elif sorted_overlappings[-1]<self.overlap_upper:
                select_new_anchor=True
                print("Reset Anchor at frame : %d by case 2.4"%f)
                continue

            # Case 2.5: No other cases:
            target_idx = sorted_idxes[-1]
            proposal_df.at[target_idx, 'personID'] = personID
            # update the dictionary of bounding box locations
            anchor_coord  = proposal_df.loc[target_idx, 'xmin':'ymax'].to_list()
            # increment f
            f+=1

        if Terminated:
            print("[WARNING] The tracking process is Terminated. Tracked data is dropped.")
            return None
        
        proposal_df = proposal_df[proposal_df.personID==personID]
        proposal_df = proposal_df[['frameID', 'xmin', 'ymin', 'xmax', 'ymax', 'personID']]
        # Interpolate and Fill NA values
        interpolated = pd.DataFrame()
        interpolated['frameID'] = range(start_frame, end_frame)

        interpolated = pd.merge(interpolated, proposal_df, on='frameID', how='left')
        interpolated['personID'] = personID
        interpolated = interpolated[~interpolated.frameID.isin(skipped_frames)].apply(lambda x: x.interpolate(method='linear'))

        # Show final confirmation
        head_at_end = self.__get_head_img(interpolated.iloc[-1].to_dict())
        window = PopupWindow(head_at_end, "[Final Check]", personID)
        decision = window.get_result()

        if decision=='No':
            print("[WARNING] Something went wrong in the middle! Unwanted head in the final. Tracked data is dropped")
            return None

        return interpolated
