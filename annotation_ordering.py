import os
import shutil

base_dir = r'D:\ShanghaiASD_project\ShanghaiASD\20230531'

def split_to_individual(grouped_by:str, delete_origin=False):
    '''
    Move files of a specific task to the individual annotation folder of each camera
    '''

    files = os.listdir('%s/%s'%(base_dir, grouped_by))
    for f in files:
        instance_id= f.split('-')[-1][:-4]
        camera_id = f[:-4]
        postfix = f[-4:]
        src = '%s/%s/%s'%(base_dir, grouped_by, f)
        dst = '%s/annotations/%s/%s/%s%s'%(base_dir, instance_id, camera_id, grouped_by, postfix)
        os.makedirs('%s/annotations/%s/%s'%(base_dir, instance_id, camera_id), exist_ok=True)
        if delete_origin:
            shutil.move(src, dst)
        else:
            shutil.copy(src, dst)
    return None
def group_all(grouped_by:str, postfix='.csv', delete_origin=False):
    '''
    Group all annotation files for a specific task into a single folder
    '''
    for instance_id in os.listdir('%s/annotations'%(base_dir)):
        for camera_id in os.listdir('%s/annotations/%s'%(base_dir, instance_id)):
            src = '%s/annotations/%s/%s/%s%s'%(base_dir, instance_id, camera_id, grouped_by, postfix)
            dst = '%s/%s/%s%s'%(base_dir, grouped_by, camera_id, postfix)
            os.makedirs('%s/%s'%(base_dir, grouped_by),exist_ok=True)
            if delete_origin:
                shutil.move(src, dst)
            else:
                shutil.copy(src, dst)

if __name__=='__main__':
    grouped_by ='gaze_points_with_patterns'
    split_to_individual(grouped_by)