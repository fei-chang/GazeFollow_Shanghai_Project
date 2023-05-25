# GazeFollow_inWild
Project Repo for handling gaze of kid and teacher during interation in an open environment.

### Current working pipeline:
- Frame Extraction
- Head Detection
- Person Identification
- Gaze Follow
- Gaze Pattern Analysis

### Project File Specifications
<details>
<summary> File Folder Structure </summary>


    ├── vids                        -> the raw videos 
    |  ├── instance_id               # each instance_id corresponding to an interaction period of a kid
    |  |  ├── camera_id.mov         
    |  |  ├── ...                   
    |  |  └── camera_1_3.mov        
    |  └── ...                    
    ├── frames                      -> the extracted frames
    |  ├── instance_id            
    |  |  ├── camera_id              # frames are named in format %06d.jpg
    |  |  |  ├── 000001.jpg       
    |  |  |  ├── 000002.jpg       
    |  |  |  ├── ...              
    |  |  |  └── 010000.jpg       
    |  |  └── ...                 
    |  └── ...                    
    ├── annotations               
    |  ├── instance_id         
    |  |  ├── camera_id    
    |  |  |  ├── raw_detections.txt         -> head bounding box detection by yolo_v3   
    |  |  |  ├── head_annotations.csv       -> cleaned head annotations with personID
    |  |  |  └── gaze_points_personID.csv   -> estimated 2D gaze points and gaze patterns of a specific person
    |  |  └── ...                 
</details>

<details>
<summary> File Format Specification </summary>

1. The raw head bounding box detection by yolo_v3: 
    - :x: No column header, entries are added as ['frameID', 'xmin', 'ymin', 'xmax', 'ymax']
    - :x: No index column
    - Entries of 'xmin', 'ymin', 'xmax', 'ymax' are all in 0-1 scale
2. The cleaned head annotations with personID: 
    - ✅Has column header: ['frameID', 'xmin', 'ymin', 'xmax', 'ymax', 'personID', 'missing']
    - :x: No index column
    - Entries of 'xmin', 'ymin', 'xmax', 'ymax' are all in 0-1 scale; 
    - Entries of 'personID' are one of values: 'kid'/'teacher'; 
    - Entries of 'missing' are booleans - with True indicating the head of the target person is undetected/occluded/outside-the-frame.
3. Estimated 2D gaze point and gaze patterns: 
    - TODO
</details>