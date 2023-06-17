
from utils import cv2_safe_read
import pandas as pd
import cv2


class Visualizer:
    def __init__(self):
        self._frame_list = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # set the codec
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.circle_thickness = 5
        self.line_thickness = 2
        self.font_size = 1

    def load_frames(self, frame_dir:str, frameIDs:enumerate, compression=0.5):
        initial_frame = cv2_safe_read('%s\\%06d.jpg'%(frame_dir, frameIDs[0])) 
        
        h, w, _ = initial_frame.shape
        h, w = map(int, [h*compression, w*compression])
        self.h = h 
        self.w = w
        for frame_num in frameIDs:
            frame = cv2_safe_read('%s\\%06d.jpg'%(frame_dir, frame_num))
            self._frame_list[frame_num] = cv2.resize(frame, (w,h)) if compression<1 else frame

    def draw_gaze_general(self, annotations:pd.DataFrame):

        grouped_df = annotations.groupby('frameID')
        for frame_num in self._frame_list.keys():
            # get_annotation
            annotations = grouped_df.get_group(frame_num)
            for idx in range(len(annotations)):
                
                info = annotations.iloc[idx]
                xmin, ymin, xmax, ymax, personID = info[['xmin', 'ymin', 'xmax', 'ymax','personID']]
                xmin, ymin, xmax, ymax = map(int, [xmin*self.w, ymin*self.h, xmax*self.w, ymax*self.h])
                color = (0,255,0) if (personID=='teacher') else (0,0,255)
                # Draw Headbox
                frame = self._frame_list[frame_num] 
                frame = cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color, self.line_thickness) # Draw head box
                
                # Write Gaze Pattern
                gaze_pattern = info['pattern']
                if gaze_pattern:
                    frame  = cv2.putText(frame, str(gaze_pattern), (xmin+5, ymax+20), self.font, self.font_size, (0,255,0), self.line_thickness)

                # Draw Sight Line
                center_x, center_y = map(int,[(xmax+xmin)/2, (ymin+ymax)/2])
                gaze_x, gaze_y =  info[['gaze_x', 'gaze_y']]
                gaze_x, gaze_y = map(int, [gaze_x*self.w, gaze_y*self.h])

                frame = cv2.circle(frame, (gaze_x, gaze_y), self.circle_thickness, color, -1) # Draw Gaze Point
                frame = cv2.line(frame, (center_x, center_y), (gaze_x, gaze_y), color, self.line_thickness) # Draw line from head center to gaze point
            self._frame_list[frame_num] = frame
        print("Finished Drawing General Gaze Signals")

    def draw_object_box(self, annotations:pd.DataFrame):
        grouped_df = annotations.groupby('frameID')
        for frame_num in self._frame_list.keys():
            # get_annotation
            annotations = grouped_df.get_group(frame_num)
            for idx in range(len(annotations)):
                info = annotations.iloc[idx]
                xmin, ymin, xmax, ymax = info[['xmin', 'ymin', 'xmax', 'ymax']]
                xmin, ymin, xmax, ymax = map(int, [xmin*self.w, ymin*self.h, xmax*self.w, ymax*self.h])
                color = (255,0,0) 
                frame = self._frame_list[frame_num] 
                frame = cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color, self.line_thickness) # Draw head box
            self._frame_list[frame_num] = frame
    
    def generate_output_vid(self, output_vid_path:str, fps = 30):
        out = cv2.VideoWriter(output_vid_path, self.fourcc, fps, (self.w, self.h))
        for frame_num in self._frame_list.keys():
            frame = self._frame_list[frame_num] 
            frame = cv2.putText(frame, '%06d'%frame_num,(50, 100), self.font, self.font_size, (0,255,0), self.line_thickness)
            # output to video
            out.write(frame) 
        out.release()

    def empty_frames(self):
        self._frame_list = {}
