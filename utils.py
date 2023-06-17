import subprocess
import numpy as np
from pathlib import Path
import pandas as pd
import os 
import cv2

def cv2_safe_read(img_path):
    '''
    To read files with Chinese characters
    '''
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

def cv2_safe_write(img, filename):
    '''
    To write files with Chinese characters
    '''
    cv2.imencode('.jpg', img)[1].tofile(filename)

def expand_headbox(head_box:list, k:float):
    '''
    expand headbox by factor k
    '''
    xmin, ymin, xmax, ymax = head_box
    updated_xmin = max(0, xmin - k*(xmax - xmin))
    updated_ymin = max(0, ymin - k*(ymax - ymin))
    updated_xmax = min(1, xmax + k*(xmax - xmin))
    updated_ymax = min(1, ymax + k*(ymax - ymin))

    return [updated_xmin, updated_ymin, updated_xmax, updated_ymax]

def frame_extraction(input_vid:str, output_dir:str, fps=30):
    '''
    Args:
    input_vid:  the path of input video
    output_dir: the directory to save extracted frames
    fps: extraction rate (frame per second), if not specified (fps=-1), will extract based on fps of the given video

    '''
    Path(output_dir).mkdir(exist_ok=True)
    
    if fps<0:
        ffmpeg_command = 'ffmpeg -i \'{}\' -vsync 0 {}/%06d.jpg'.format(input_vid, output_dir)
    else:
        ffmpeg_command = 'ffmpeg -i {} -r {} -q:v 2 -f image2 {}/%06d.jpg'.format(input_vid, fps, output_dir)

    subprocess.run(ffmpeg_command, shell=True)

def visualize(output_vid: str, frame_dir:str, annotations:str, 
              gaze_heatmaps=True, gaze_points = True, gaze_patterns = True,
              fps = 30, rate=1, compression=1, save_img = False):
    '''
    Args:
    output_vid:     Path of the visualization video
    frame_dir:      Directory to read in frames
    annotations:    Path of the annotation file to visualize with
    gaze_heatmaps:  Whether or not to visualize gaze heatmaps
    gaze_points:    Whether or not to visualize the 2D gaze points
    gaze_patterns:  Whehter or not to visualize the gaze patterns
    fps:            Play speed of the visualization video (frame per second)
    rate:           The select rate of frames to visualize with (*for faster implementation)
    compression:    The compression rate of frames (*for faster implementation)
    save_img:       Whether or not save visualization image

    '''
    ############################################################################################
    # Input Check
    source_labels = pd.read_csv(annotations)
    visualization_labels = source_labels[source_labels.frameID%rate==0]

    grouped_df = visualization_labels.groupby('frameID')
    valid_frames = [int(f[:-4]) for f in os.listdir(frame_dir)]
    assert len(valid_frames)>0
    for frame_num in visualization_labels.frameID.unique():
        assert frame_num in valid_frames

    initial_frame = cv2_safe_read('%s\\%06d.jpg'%(frame_dir, valid_frames[0])) 
    h, w, _ = initial_frame.shape
    h, w = map(int, [h*compression, w*compression])
    if save_img:
        # Create output path
        out_path = output_vid.split('.')[0]
        os.makedirs(out_path, exist_ok=True)

    print("Visualization Starts")
    # Initialize output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # set the codec
    out = cv2.VideoWriter(output_vid, fourcc, fps, (w,h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    circle_thickness = 5
    line_thickness = 2
    font_size = 1

    for frame_num in visualization_labels.frameID.unique():
        # read_frame
        frame = cv2_safe_read('%s\\%06d.jpg'%(frame_dir, frame_num))
        frame = cv2.resize(frame, (w,h)) if compression<1 else frame

        # get_annotation
        annotations = grouped_df.get_group(frame_num)
        for idx in range(len(annotations)):
            info = annotations.iloc[idx]
            xmin, ymin, xmax, ymax, personID = info[['xmin', 'ymin', 'xmax', 'ymax','personID']]
            xmin, ymin, xmax, ymax = map(int, [xmin*w, ymin*h, xmax*w, ymax*h])
            color = (0,255,0) if (personID=='teacher') else (0,0,255)
            frame = cv2.rectangle(frame, (xmin,ymin), (xmax, ymax), color, line_thickness) # Draw head box
            
            if gaze_heatmaps:
                #TODO:
                print("Gaze heatmap visualization Under Construction")
            if gaze_patterns:
                gaze_pattern = info['pattern']
                if gaze_pattern:
                    frame = cv2.putText(frame, str(gaze_pattern), (xmin+5, ymax+20), font, font_size, (0,255,0), line_thickness)

            if gaze_points:
                center_x, center_y = map(int,[(xmax+xmin)/2, (ymin+ymax)/2])
                gaze_x, gaze_y =  info[['gaze_x', 'gaze_y']]
                gaze_x, gaze_y = map(int, [gaze_x*w, gaze_y*h])

                frame = cv2.circle(frame, (gaze_x, gaze_y), circle_thickness, color, -1) # Draw Gaze Point
                frame = cv2.line(frame, (center_x, center_y), (gaze_x, gaze_y), color, line_thickness) # Draw line from head center to gaze point
        if save_img:
            # save to img
            cv2_safe_write(frame, '%s/%06d.jpg'%(out_path, frame_num))

        frame = cv2.putText(frame, '%06d'%frame_num,(50, 100), font, font_size, (0,255,0), line_thickness)
        # output to video
        out.write(frame) 

    out.release()
    return 
