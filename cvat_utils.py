import xml.etree.ElementTree as ET
import numpy as np

def cvat2dict(cvat_file_path, box_label='head', point_label='gaze'):
    '''
    Convert the CVAT task annotation file to a dictionary.
    Args:
    cvat_file_path: The path to exported cvat annotation file xml.
    box_label:      A string contained in all box annotations. Used as an identifier to get bounding box annotations.
    point_label:    A string contained in all point annotations. Used as an identifier to get point-like annotations.

    Returns:
    info_dict:      A dictionary with dict[task_name][frameID][annotation_label] containing value of a annotation at a specific frame.
    '''

    info_dict = {}

    tree = ET.parse(cvat_file_path)
    # get the root element
    root = tree.getroot()

    for task in root.iter(tag='task'): # Iterate over task
        task_name = task.find('name').text
        print("[INFO] Converting Format for Task %s"%(task_name))
        task_vid_name = task.find('source').text
        print('[INFO] The original video is %s'%task_vid_name)
        task_id = task.find('id').text
        info_dict[task_name] = {}
        width = int(task.find('original_size').find('width').text)
        height = int(task.find('original_size').find('height').text)

        for tr in root.iterfind(".//track[@task_id='%s']"%task_id):
            label = tr.get('label')
            if box_label in label: # the annotation is a head bounding box
                for head in tr.iter(tag='box'):
                    frame = int(head.get('frame'))
                    if not frame in info_dict:
                        info_dict[task_name][frame] = {}
                    xmin, ymin, xmax, ymax = map(float, [head.get('xtl'), head.get('ytl'), head.get('xbr'), head.get('ybr')])
                    info_dict[task_name][frame][label] = [xmin/width, ymin/height, xmax/width, ymax/height] # Normalize box coordinates to scale 0-1
                    
            elif point_label in label: # the annotation is a gaze point
                for point in tr.iter(tag='points'):
                    frame = int(point.get('frame'))
                    if not frame in info_dict:
                        info_dict[task_name][frame] = {}
                    gaze_x, gaze_y = np.array(point.get('points').split(",")).astype(float)
                    info_dict[task_name][frame][label] = [gaze_x/width, gaze_y/height] # Normalize gaze to scale 0-1
    
    return info_dict
