
from utils import cv2_safe_read
import pandas as pd
import cv2
import numpy as np
import json

class Visualizer:
    def __init__(self):
        self._frame_list = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # set the codec
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.circle_thickness = 5
        self.line_thickness = 2
        self.font_size = 1
        self.teacher_color = (0,255,0) # green
        self.student_color = (0,0,255) # red
        self.object_color = (255,0,0) # blue

    def load_frames(self, frame_dir:str, frameIDs:enumerate, compression=0.5):
        initial_frame = cv2_safe_read('%s\\%06d.jpg'%(frame_dir, frameIDs[0])) 
        h, w, _ = initial_frame.shape
        h, w = map(int, [h*compression, w*compression])
        self.h = h 
        self.w = w
        for frame_num in frameIDs:
            frame = cv2_safe_read('%s\\%06d.jpg'%(frame_dir, frame_num))
            self._frame_list[frame_num] = cv2.resize(frame, (w,h)) if compression<1 else frame

    def draw_gaze_general(self, annotations:pd.DataFrame, show_illustration=False):
        fig_path = r'D:\ShanghaiASD_project\ShanghaiASD\Misc'
        grouped_df = annotations.groupby('frameID')
        for frame_num in annotations.frameID.unique():
            # get_annotation
            annotations = grouped_df.get_group(frame_num)
            for idx in range(len(annotations)):
                
                info = annotations.iloc[idx]
                xmin, ymin, xmax, ymax, personID = info[['xmin', 'ymin', 'xmax', 'ymax','personID']]
                xmin, ymin, xmax, ymax = map(int, [xmin*self.w, ymin*self.h, xmax*self.w, ymax*self.h])
                color = self.teacher_color if (personID=='teacher') else self.student_color


                # Draw Headbox
                frame = self._frame_list[frame_num] 
                frame = cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color, self.line_thickness) # Draw head box
                
                # Write Gaze Pattern
                gaze_pattern = info['pattern']
                if type(gaze_pattern) is str:
                    frame  = cv2.putText(frame, str(gaze_pattern), (xmin+5, ymax+20), self.font, self.font_size, color, self.line_thickness)
                    if (show_illustration&(personID=='student')): # Draw based on student's gaze pattern
                        fig = cv2_safe_read('%s/%s_figure.png'%(fig_path, gaze_pattern.lower()))
                        fig_h, fig_w, _ = fig.shape
                        frame[:fig_h, -fig_w:, :] = fig
                        frame = cv2.rectangle(frame, (self.w-fig_w, 0), (self.w, fig_h), color, self.line_thickness)
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
        for frame_num in annotations.frameID.unique():
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
    
    def load_emotion(self, emotion_dir):
        self.label_list = pd.read_csv(emotion_dir,sep=',',header=None).values
        self.neutral_list=[]
        self.angry_list=[]
        self.disgust_list=[]
        self.fear_list=[]
        self.happy_list=[]
        self.sad_list=[]
        self.surprise_list=[]
        last_item = 8*['0.143']
        for i in range(len(self.label_list)):
            item = self.label_list[i]
            if 'frameID' not in item:
                if type(item[1]) == float: #遇到null都置0
                    item = last_item
                self.neutral_list.append(float(item[1]))
                self.angry_list.append(float(item[2]))
                self.disgust_list.append(float(item[3]))
                self.fear_list.append(float(item[4]))
                self.happy_list.append(float(item[5]))
                self.sad_list.append(float(item[6]))
                self.surprise_list.append(float(item[7]))
                last_item = item

    def draw_emotion_curve(self,bar_height=90,font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=0.5):
        color_list=[(255,0,0),(255,165,0),(128,128,0),(0,255,0),(0,191,243),(0,0,255),(233,0,233)]
        #draw curve
        for frame_num in self._frame_list.keys():
            i = frame_num-1
            img = self._frame_list[frame_num]

            #curve param
            curve_h = bar_height
            curve_thick = 2

            #padding
            img = cv2.copyMakeBorder(img,0,curve_h+10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            im_h = img.shape[0]
            im_w = img.shape[1]
            self.h = im_h
            self.w = im_w

            #draw curve
            word_w = 140
            space = (im_w-word_w)/len(self._frame_list.keys())

            for j in range(i):
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.neutral_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.neutral_list[j+1]*curve_h)-5),color_list[0],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.angry_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.angry_list[j+1]*curve_h)-5),color_list[1],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.disgust_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.disgust_list[j+1]*curve_h)-5),color_list[2],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.fear_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.fear_list[j+1]*curve_h)-5),color_list[3],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.happy_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.happy_list[j+1]*curve_h)-5),color_list[4],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.sad_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.sad_list[j+1]*curve_h)-5),color_list[5],curve_thick)
                cv2.line(img,(int(j*space)+word_w,im_h-int(self.surprise_list[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.surprise_list[j+1]*curve_h)-5),color_list[6],curve_thick)
            
            #为了不让字出格子，使其逐渐往左偏移的宽度（像素数），最大70
            all_num=len(self.neutral_list) #所有帧的数量
            shift_wid = int(70.0*(frame_num-all_num*0.8)/(all_num*0.2)) if frame_num > all_num*0.8 else 0

            cv2.putText(img,'neutral',(int((i)*space)+word_w-shift_wid,im_h-int(self.neutral_list[i]*curve_h)-5),font,font_scale,color_list[0],2)
            cv2.putText(img,'angry',(int((i)*space)+word_w-shift_wid,im_h-int(self.angry_list[i]*curve_h)-5),font,font_scale,color_list[1],2)
            cv2.putText(img,'disgust',(int((i)*space)+word_w-shift_wid,im_h-int(self.disgust_list[i]*curve_h)-5),font,font_scale,color_list[2],2)
            cv2.putText(img,'fear',(int((i)*space)+word_w-shift_wid,im_h-int(self.fear_list[i]*curve_h)-5),font,font_scale,color_list[3],2)
            cv2.putText(img,'happy',(int((i)*space)+word_w-shift_wid,im_h-int(self.happy_list[i]*curve_h)-5),font,font_scale,color_list[4],2)
            cv2.putText(img,'sad',(int((i)*space)+word_w-shift_wid,im_h-int(self.sad_list[i]*curve_h)-5),font,font_scale,color_list[5],2)
            cv2.putText(img,'surprise',(int((i)*space)+word_w-shift_wid,im_h-int(self.surprise_list[i]*curve_h)-5),font,font_scale,color_list[6],2)

            cv2.putText(img,'Emotion',(30,im_h-int(curve_h/2)),font,font_scale,(0,0,0),2)
            self._frame_list[frame_num] = img
    
    def draw_focus_curve(self,bar_height=90,teacher_curve_color=(0,255,0),student_curve_color=(0,0,255),font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=0.5):
        #draw curve
        for frame_num in self._frame_list.keys():
            i = frame_num-1
            img = self._frame_list[frame_num]

            #curve param
            curve_h = bar_height
            curve_thick = 2

            #padding
            img = cv2.copyMakeBorder(img,0,curve_h*2+10,0,0,cv2.BORDER_CONSTANT,value=[255,255,255])
            im_h = img.shape[0]
            im_w = img.shape[1]
            self.h = im_h
            self.w = im_w

            #draw curve
            word_w = 140
            space = (im_w-word_w)/len(self._frame_list.keys())

            for j in range(i):
                #mark if the student is looking at the camera
                if self.prob_list_student[j]>0.5 and self.prob_list_student[j+1]>0.5:
                    cv2.fillConvexPoly(img,
                    np.array([[int(j*space)+word_w,im_h-curve_h-5],[int(j*space)+word_w,im_h-5],[int((j+1)*space)+word_w,im_h-5],[int((j+1)*space)+word_w,im_h-curve_h-5]]),
                    (200,200,255))

                #student curve
                if self.prob_list_student[j]>0 and self.prob_list_student[j+1]>0:
                    cv2.line(img,(int(j*space)+word_w,im_h-int(self.prob_list_student[j]*curve_h)-5),(int((j+1)*space)+word_w,im_h-int(self.prob_list_student[j+1]*curve_h)-5),student_curve_color,curve_thick)

                #mark if the teacher is looking at the camera
                if self.prob_list_teacher[j]>0.5 and self.prob_list_teacher[j+1]>0.5:
                    cv2.fillConvexPoly(img,
                    np.array([[int(j*space)+word_w,im_h-curve_h*2-10],[int(j*space)+word_w,im_h-curve_h-10],[int((j+1)*space)+word_w,im_h-curve_h-10],[int((j+1)*space)+word_w,im_h-curve_h*2-10]]),
                    (200,255,200))

                #teacher curve
                if self.prob_list_teacher[j]>0 and self.prob_list_teacher[j+1]>0:
                    cv2.line(img,(int(j*space)+word_w,im_h-int(self.prob_list_teacher[j]*curve_h)-curve_h-10),(int((j+1)*space)+word_w,im_h-int(self.prob_list_teacher[j+1]*curve_h)-curve_h-10),teacher_curve_color,curve_thick)

            #the probability=0.5 line
            cv2.line(img,(word_w,im_h-int(curve_h/2)),(int(i*space)+word_w,im_h-int(curve_h/2)),student_curve_color,1)
            cv2.line(img,(word_w,im_h-int(curve_h/2)-curve_h-5),(int(i*space)+word_w,im_h-int(curve_h/2)-curve_h-5),teacher_curve_color,1)

            #text
            cv2.putText(img,'probability=0.5',(word_w,im_h-int(curve_h/2)-15),font,font_scale,student_curve_color,1)
            cv2.putText(img,'probability=0.5',(word_w,im_h-int(curve_h/2)-curve_h-15),font,font_scale,teacher_curve_color,1)
            cv2.putText(img,'teacher looking',(0,im_h-int(curve_h/2)-curve_h-15),font,font_scale,teacher_curve_color,2)
            cv2.putText(img,'at the camera',(0,im_h-int(curve_h/2)-curve_h+15),font,font_scale,teacher_curve_color,2)
            cv2.putText(img,'student looking',(0,im_h-int(curve_h/2)-15),font,font_scale,student_curve_color,2)
            cv2.putText(img,'at the camera',(0,im_h-int(curve_h/2)+15),font,font_scale,student_curve_color,2)

            self._frame_list[frame_num] = img
    
    def _dict2list(self,in_dict):
        out_list = []
        for i in range(1,len(self._frame_list.keys())+1):
            if str(i) in in_dict:
                out_list.append(in_dict[str(i)])
            else:
                out_list.append(-1)
        return out_list

    def load_focus_prob(self, student_prob_dir, teacher_prob_dir):
        #get the probability of looking at the camera from the json file
        with open(student_prob_dir,'r') as f:
            prob_dict = json.load(f)
        self.prob_list_student = self._dict2list(prob_dict)

        with open(teacher_prob_dir,'r') as f:
            prob_dict = json.load(f)
        self.prob_list_teacher = self._dict2list(prob_dict)

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
