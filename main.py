import argparse
import os
from utils import frame_extraction, visualize
from GazeHandler import GazeHandler

def get_args_parser():
    parser = argparse.ArgumentParser('General Setup', add_help=False)
    # Setup Path
    parser.add_argument('--base_dir', type=str, default = '')
    parser.add_argument('--model_weights', type=str, default = '/home/changfei/gaze_follow/model_gazefollow.pt')
    
    # Setup Instance and Camera ids
    parser.add_argument('--instance_id', type=str, default='all')
    parser.add_argument('--camera_id', type=str, default='all')
    # Stage selection
    parser.add_argument('--extract_frames', action="store_true")
    parser.add_argument('--detect_heads', action="store_true")
    parser.add_argument('--identify_person', action="store_true")
    parser.add_argument('--obtain_gaze', action='store_true')
    parser.add_argument('--visualize_only_head', action='store_true')
    parser.add_argument('--visualize_all', action='store_true')

    parser.add_argument('--full_pipeline', action='store_true')
def main(args):
    # Confirm Info 
    print("Running Gaze Follow and Gaze Pattern pipeline...")
    print("Base Directory: %s"%args.base_dir)

    if args.obtain_gaze or args.full_pipeline:
        gaze_handler = GazeHandler(args.model_weights)
    
    instances =  os.listdir(args.base_dir) if args.instance_id=='all' else [args.instance_id]
    for instance_id in instances:
        cameras = os.listdir('%s/%s'%args.base_dir, instance_id) if args.camera_id=='all' else [args.camera_id]
        for camera_id in cameras:
            # Setting up directories
            raw_vid = '%s/%s/%s.mov'%(args.base_dir, instance_id, camera_id)
            assert os.path.exists(raw_vid)

            frame_dir = '%s/frames/%s/%s'%(args.base_dir, instance_id, camera_id)
            os.path.makedirs(frame_dir, exist_ok=True)
            
            annotation_dir = '%s/annotations/%s/%s'%(args.base_dir, instance_id, camera_id)
            os.path.makedirs(annotation_dir, exist_ok=True)

            visualization_dir = '%s/visualize'
            os.path.makedirs(visualization_dir, exist_ok=True)

            raw_detections = '%s/raw_detections.txt'%(annotation_dir)
            head_boxes_file = '%s/head_annotations.csv'%(annotation_dir)
            heatmap_pkl = '%s/heatmap'%(annotation_dir)
            gaze_file = '%s/gaze_points_with_patterns.csv'%(annotation_dir)


            if args.extract_frames or args.full_pipeline:
                # Perform Frame Extraction
                print("Extracting Frames for %s, %s ..."%(instance_id, camera_id), end = '\t'*4)
                frame_extraction(raw_vid, frame_dir)
            if args.detect_heads or args.full_pipeline:
                print('TODO')
                # print("Detecting Heads under Construction...")
            if args.identify_person or args.full_pipeline:
                print('TODO')
                # print("Identifying person under Construction...")
            if args.visualize_only_head or args.full_pipeline: 
                print('TODO')
            if args.obtian_gaze or args.full_pipeline:
                print("Running Gaze Follow and Gaze Pattern...")
                assert(len(os.listdir(args.frame_dir))>0)
                assert(os.path.exists(head_boxes_file))
                gaze_handler.process(head_boxes_file, frame_dir, heatmap_pkl, gaze_file)

            if args.visualize_all or args.full_pipeline:
                output_vid = '%s/%s_%s'%(visualization_dir, instance_id, camera_id)
                assert(len(os.listdir(args.frame_dir))>0)
                assert(os.path.exists(gaze_file))
                visualize(output_vid, frame_dir, gaze_file, gaze_heatmaps=False, compression=0.5)
