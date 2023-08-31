# GazeFollow_inWild
Project Repo for handling gaze of kid and teacher during interation in Shanghai Project Settings.


### Running Instruction
`
python main.py --base_dir base_dir --instance_id instance_id --camera_id camera_id --full_pipeline
`
### Current working pipeline
- Frame Extraction
- Head Detection and Tracking
- Person Identification
- Gaze Follow and Gaze Pattern

### 文件存储
<details>
<summary> 文件结构 </summary>
    ├── vids                        -> 视频
    |  ├── instance_id(kid)           # 小孩子的ID
    |  |  ├── camera_id.mov/mp4         
    |  |  ├── ...                   
    |  |  └── camera_1_3.mov/mp4        
    |  └── ...                    
    ├── frames                      -> 视频帧
    |  ├── instance_id            
    |  |  ├── camera_id              # 视频命名规则：%06d.jpg
    |  |  |  ├── 000001.jpg       
    |  |  |  ├── 000002.jpg       
    |  |  |  ├── ...              
    |  |  |  └── 010000.jpg       
    |  |  └── ...                 
    |  └── ...                    
    ├── head_annotations               
    |  ├── instance_id         
    |  |  ├── camera_id.csv
    |  |  └── ...                 
</details>

<details>
<summary> 文件格式说明 </summary>
1. **head_annotations** 头部标注
    - ✅Has column header: `['frameID', 'xmin', 'ymin', 'xmax', 'ymax', 'personID', 'activity_split']`
    - :x: No index column
    - Entries of 'xmin', 'ymin', 'xmax', 'ymax' are all normalized in 0-1 scale; 
    - Entries of 'personID' are one of values: 'kid'/'teacher'; 
    - Entries of activity split associates to the activity of the kid and teacher.
3. **gaze_points_with_patterns.csv** (Estimated 2D gaze point and gaze patterns)
    - :white_check_mark: Has column header: `['frameID', 'personID', 'xmin', 'ymin', 'xmax', 'ymax', 'gaze_x', 'gaze_y', 'pattern', 'missing']`
    - :x: No index column
    - Entries of 'xmin', 'ymin', 'xmax', 'ymax', 'gaze_x', 'gaze_y' are all in 0-1 scale; 
    - Entries of 'pattern' contains values of 'Share', 'Mutual', 'Single', 'Miss', 'Void' and None for frames where only one person is present.
</details>
