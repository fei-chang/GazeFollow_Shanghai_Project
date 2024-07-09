import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def collect_lists(gt_df, pred_df):
    gt_ls = []
    for i in gt_df.index:
        start_f = gt_df.at[i, 'start_frame']
        end_f = gt_df.at[i, 'end_frame']
        for f in range(start_f, end_f):
            gt_ls.append(f)
    
    pred_ls = []
    for j in pred_df.index:
        if 'teacher' in pred_df.at[j, 'pattern']:
            f = pred_df.at[j, 'frameID']
            pred_ls.append(f)
    return gt_ls, pred_ls
            
def obtain_score(a_pred, a_gt, total_frames):
    gt_ls, pred_ls = collect_lists(a_gt, a_pred)        
    tp_ls = []
    fp_ls = []
    fn_ls = []
    tn_ls = []
    for f in total_frames:
        if f in pred_ls: # a predicted positive
            if f in gt_ls: # true positive
                tp_ls.append(f)
            else: # false positive
                fp_ls.append(f)
        else: # a predicited negative
            if f in gt_ls: # false negative
                fn_ls.append(f)
            else: # true negative
                tn_ls.append(f)
    assert len(tp_ls)+len(fp_ls)+len(fn_ls)+len(tn_ls)==len(total_frames)
    acc_score = (len(tp_ls)+len(tn_ls))/len(total_frames)
    TP = len(tp_ls)
    TN = len(tn_ls)
    FN = len(fn_ls)
    FP = len(fp_ls)


    if len(gt_ls)==0:
        recall = np.nan
    else:
        recall = len(tp_ls)/(len(tp_ls)+len(fn_ls))
    if len(pred_ls)==0:
        precision = np.nan
    else:
        precision = len(tp_ls)/(len(tp_ls)+len(fp_ls))
    
    Sensitivity = TP / (TP + FN)
    Specificity =TN / (TN + FP)
    balanced_acc = (Sensitivity+Specificity)/2
    return acc_score, balanced_acc ,recall, precision



cleaned_gt_dir = r'D:\ShanghaiASD_project\gazefollow_pattern_checkup\ENCU_base_annotations\cleaned'
activity_dir = r'D:\ShanghaiASD_project\gazefollow_pattern_checkup\activity_mapping'
results_dir = r'D:\ShanghaiASD_project\gazefollow_pattern_checkup\results_v1127'


def compute_old(kid_id):
    pred_files_dir =  r'D:\ShanghaiASD_project\gazefollow_pattern_checkup\pred_stats\20230803_v0918'
    pred_dir = '%s/%s'%(pred_files_dir, kid_id)

    a_file = '%s/%s_activity.csv'%(activity_dir, kid_id)
    a_df = pd.read_csv(a_file)

    gt_df = pd.read_csv('%s/%s.txt'%(cleaned_gt_dir, kid_id))

    stats_header = ['kid', 'camera', 'acc', 'balanced_acc', 'recall', 'precision']
    pd_stats = []
    for cam in os.listdir(pred_dir):
        pred_cam_df = pd.read_excel('%s/%s/activity_summary_kid.xlsx'%(pred_dir, cam))
        gt_kid_at_teacher = gt_df[gt_df.SXC=='face'].dropna(subset=['activity'])
        total_frames = []
        for i in range(len(a_df)):
            start_f, end_f, act = a_df.iloc[i]
            for i in range(start_f, end_f):
                total_frames.append(i)
        # a_gt = gt_kid_at_teacher[gt_kid_at_teacher.activity==act]
        # a_pred = pred_cam_df[pred_cam_df.activity==act]

        acc_score, balanced_acc, recall, precision = obtain_score(pred_cam_df, gt_kid_at_teacher, total_frames)
        # print("In camera: %s, Acc: %.2f, Recall: %.2f, Precision: %.2f"%(cam_name, acc_score*100, recall*100, precision*100))
        pd_stats.append([kid_id, cam, acc_score*100, balanced_acc*100, recall*100, precision*100])


    final_df = pd.DataFrame(pd_stats, columns=stats_header)
    return final_df
    # total_stats  = 

kid_id = '109'
dfs = []
for kid_id in ['041','042', '062', '109']:
    kid_df = compute_old(kid_id)
    dfs.append(kid_df)
final_df = pd.concat(dfs)
final_df.to_excel('%s/old_1127.xlsx'%results_dir, index=False)