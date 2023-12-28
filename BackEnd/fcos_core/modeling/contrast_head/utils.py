import numpy as np
import torch
import random
from scipy.ndimage.morphology import distance_transform_edt
import torch.nn.functional as F
random.seed(1)

def enhance_op(x):
    assert x.dim() == 4, f'only support BxCxHxW which dim=4'
    N, C, H, W = x.shape
    phi_x = F.normalize(x, p=2, dim=1, eps=1e-12, out=None).view(N, C, -1)
    theta_x = F.normalize(x, p=2, dim=1, eps=1e-12, out=None).view(N, C, -1).permute(0,2,1)
    pairwise_weight = torch.matmul(theta_x, phi_x)
    pairwise_weight = pairwise_weight.softmax(dim=-1)
    x = x.view(N, C, -1).permute(0, 2, 1)
    out = torch.matmul(pairwise_weight, x)  
    out = out.permute(0, 2, 1).contiguous().reshape(N, C, H, W)

    return out

def normalize_batch(cams_batch):
    """
    Classic min-max normalization
    """
    bs = cams_batch.size(0)
    cams_batch = cams_batch + 1e-4
    cam_mins = getattr(cams_batch.view(bs, -1).min(1), 'values').view(bs, 1, 1, 1)
    cam_maxs = getattr(cams_batch.view(bs, -1).max(1), 'values').view(bs, 1, 1, 1)
    return (cams_batch - cam_mins) / (cam_maxs - cam_mins)


def get_query_keys_eval(cams):
    """
        Input
            cams: Tensor, cuda, Nx1x28x28

        Here, when performing evaluation, only cams are provided for all categories, including base and novel.
    """
    return_result = dict()
    cams  = torch.tensor(cams)
    cams = normalize_zero_to_one(cams) #tensor  shape:N,28,28, 0-1

    # we only need queries
    query_pos_sets = torch.where(cams>0.92, 1.0, 0.0).to(dtype=torch.bool)
    query_neg_sets = torch.where(cams<0.08, 1.0, 0.0).to(dtype=torch.bool)
    
    return_result['query_pos_sets'] = query_pos_sets
    return_result['query_neg_sets'] = query_neg_sets

    return return_result

def get_query_keys( cams,
                    pt_map,
                    bkg,
                    thred_u=0.2,
                    scale_u=1.0,
                    percent=0.3):
    """
        Input
            cams: Tensor, cuda, Nx1x28x28
            edges: Tensor, cuda, Nx28x28
            masks: Tensor, cuda, Nx28x28 
    """
    #######################################################
    #---------- some pre-processing -----------------------
    #######################################################
    pt_map = torch.stack(pt_map).squeeze(1).cpu()
    bkg = torch.stack(bkg).squeeze(1).cpu()
    cams  = torch.tensor(cams)
    cams = normalize_zero_to_one(cams) #normalize to 0~1

   
    #######################################################
    #---------- get query mask for each proposal ----------
    #######################################################
    query_pos_sets = pt_map.to(dtype=torch.bool)  #here, pos=foreground area  neg=background area
    query_neg_sets = bkg.to(dtype=torch.bool)

    # if available points in a mask less than 2. replace it with cam. Note cam is available for base(seen) and novel(unseen)
    keep_pos_flag = torch.where(query_pos_sets.sum(dim=[1,2])<2.0, 1.0, 0.0).to(dtype=torch.bool)
    keep_neg_flag = torch.where(query_neg_sets.sum(dim=[1,2])<2.0, 1.0, 0.0).to(dtype=torch.bool)
    if True in keep_pos_flag:
        cam_pos_high = torch.where(cams>0.9, 1.0, 0.0).to(dtype=torch.bool)
        query_pos_sets[keep_pos_flag] = cam_pos_high[keep_pos_flag] #replace pos-mask with high-confidence (0.95) cam
    if True in keep_neg_flag:
        cam_neg_high = torch.where(cams<0.1, 1.0, 0.0).to(dtype=torch.bool)
        query_neg_sets[keep_neg_flag] = cam_neg_high[keep_neg_flag] #replace neg-mask with low-confidence (0.05) cam

    #######################################################
    # ----------- get different types of keys -------------
    #######################################################
    
    positive_sets = torch.where(cams > (1.0 - thred_u*scale_u), 1.0, 0.0) #scale_u can adjust the threshold, it is not used in our paper.
    negative_sets = torch.where(cams < (thred_u*scale_u), 1.0, 0.0)
    #######################################################
    # --------- determine the number of sampling ----------
    #######################################################
    # how many points can be sampled for all proposals for each type of sets
    num_pos = positive_sets.sum(dim=[1,2]) #E=easy, H=hard
    num_neg = negative_sets.sum(dim=[1,2])

    # if available points are less then 5 for each type, this proposal will be dropped out.
    available_num = torch.cat([num_pos, num_neg])
    abandon_inds = torch.where(available_num < 5, 1, 0).reshape(2, -1) 
    keeps = torch.logical_not(abandon_inds.sum(0).to(dtype=torch.bool))
    
    if True not in keeps:  # all proposals do not have enough points that can be sample. This is a extreme situation.
        # set the points number of all types sets to 2
        # sometimes, there would still raise an error. I will fix it later.
        sample_num_pos = torch.ones_like(num_pos) * 2
        sample_num_neg = torch.ones_like(num_neg) * 2
        print('[sample points]:{}'.format(available_num))  #print log so that we can debug it.....
    else:
        sample_num_pos = (percent * num_pos[keeps]).ceil()
        sample_num_neg = (percent * num_neg[keeps]).ceil()


    #######################################################
    # ----------------- sample points ---------------------
    #######################################################
    positive_sets_N = get_pixel_sets_N(positive_sets[keeps], sample_num_pos)
    negative_sets_N = get_pixel_sets_N(negative_sets[keeps], sample_num_neg)
    # Record points number
    num_per_type = dict()
    num_per_type['pos_num_'] = sample_num_pos
    num_per_type['neg_num_'] = sample_num_neg


    #######################################################
    # ------------------- return data ---------------------
    #######################################################
    return_result = dict()
    return_result['keeps'] = keeps  #which proposal is preserved
    return_result['num_per_type'] = num_per_type
    return_result['query_pos_sets'] = query_pos_sets  # query area for foreground
    return_result['query_neg_sets'] = query_neg_sets  # query area for background
    return_result['positive_sets_N'] = positive_sets_N.to(dtype=torch.bool) 
    return_result['negative_sets_N'] = negative_sets_N.to(dtype=torch.bool) 
    return return_result

def normalize_zero_to_one(imgs):
    if isinstance(imgs, torch.Tensor):
        bs, h, w = imgs.shape
        imgs_mins = getattr(imgs.view(bs, -1).min(1), 'values').view(bs, 1, 1)
        imgs_maxs = getattr(imgs.view(bs, -1).max(1), 'values').view(bs, 1, 1)
        return (imgs - imgs_mins) / (imgs_maxs - imgs_mins)
    else:
        raise TypeError(f'Only tensor is supported!')

def get_pixel_sets_N(src_sets, select_num):
    return_ = []
    if isinstance(src_sets, torch.Tensor):
        bs, h,w = src_sets.shape
        keeps_all = torch.where(src_sets>0.5, 1, 0).reshape(bs,-1)
        for idx,keeps in enumerate(keeps_all):
            keeps_init = np.zeros_like(keeps)
            src_set_index = np.arange(len(keeps))
            src_set_index_keeps = src_set_index[keeps.numpy().astype(np.bool)]
            resultList=random.sample(range(0,len(src_set_index_keeps)),int(select_num[idx]))
            src_set_index_keeps_select = src_set_index_keeps[resultList]
            keeps_init[src_set_index_keeps_select]=1
            return_.append(keeps_init.reshape(h,w))
    else:
        raise ValueError(f'only tensor is supported!')
    return_ = np.stack(return_)
    return torch.tensor(return_) * src_sets