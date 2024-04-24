import numpy as np
import scipy.io
from tqdm import tqdm
import argparse
import os
#Added pickle to unpickle stored idx for te/tr
import pickle

def main():
    tr_file = open('tr_idx', 'rb')
    te_file = open('te_idx', 'rb')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--featname",
        type=str,
        default='',
        help="Target variable",
    )
    parser.add_argument(
        "--use_stim",
        type=str,
        default='',
        help="ave or each",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject=opt.subject
    use_stim = opt.use_stim
    featname = opt.featname
    topdir = '../../nsdfeat/'
    savedir = f'{topdir}/subjfeat/'
    featdir = f'{topdir}/{featname}/'

    # nsd_expdesign = scipy.io.loadmat('../../nsd/nsddata/experiments/nsd/nsd_expdesign.mat')

    # Note that most of them are 1-base index!
    # This is why I subtract 1
    # sharedix = nsd_expdesign['sharedix'] -1 

    #ADDED HERE
    tr_idx = pickle.load(tr_file)
    te_idx = pickle.load(te_file)

    if use_stim == 'ave':
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims_ave.npy')
    else: # Each
        stims = np.load(f'../../mrifeat/{subject}/{subject}_stims.npy')
    
    feats = []
    tr_idx_flag = np.zeros(len(stims))
    #CHANGED TO USE UNPICKLED LISTS
    for idx, s in tqdm(enumerate(stims)): 
        if s in te_idx:
            tr_idx_flag[idx] = 0
        else:
            tr_idx_flag[idx] = 1    
        feat = np.load(f'{featdir}/{s:06}.npy')
        feats.append(feat)

    feats = np.stack(feats)    

    os.makedirs(savedir, exist_ok=True)

    feats_tr = feats[tr_idx_flag==1,:]
    feats_te = feats[tr_idx_flag==0,:]
    np.save(f'../../mrifeat/{subject}/{subject}_stims_tridx.npy',tr_idx_flag)

    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_tr.npy',feats_tr)
    np.save(f'{savedir}/{subject}_{use_stim}_{featname}_te.npy',feats_te)


if __name__ == "__main__":
    main()
