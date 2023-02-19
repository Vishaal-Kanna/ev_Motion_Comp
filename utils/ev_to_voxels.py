#!/usr/bin/evn python

import numpy as np
import torch

h = 180
w = 240

def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(h, w), clip_out_of_range=True,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs*mask, ys*mask
            img.index_put_((ys, xs), ps.float(), accumulate=True)
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
            raise e
    return img

def events_to_ts_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(h, w), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)

    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.floor(t_norm)==bi #torch.max(zeros, 1.0-torch.abs(t_norm-bi)) #
            weights = ps*bilinear_weights*t_norm
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False)
        bins.append(vb)
    return bins, dt

def events_to_neg_pos_ts_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(h, w), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Two voxel grids, one for positive one for negative events
    """
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>-100, ones_v, zero_v)
    neg_weights = torch.where(ps>-100, ones_v, zero_v)
    pos_neg_img = np.zeros((9,2,h,w))
    pos_neg_img_2d = np.zeros((9,h, w))

    voxel_pos, dt = events_to_ts_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg, _ = events_to_ts_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    for i,pos in enumerate(voxel_pos):
        pos_neg_img[i,0,:,:] = pos
        pos_neg_img_2d[i,:,:] = pos
    for i,neg in enumerate(voxel_neg):
        pos_neg_img[i,1,:,:] = neg
    return pos_neg_img, pos_neg_img_2d, dt

def events_to_n_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(h, w), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Voxel of the events between t0 and t1
    """
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_norm = (ts-ts[0])/dt*(B-1)
    for bi in range(B):
        if temporal_bilinear:
            bilinear_weights = torch.floor(t_norm)==bi #torch.max(zeros, 1.0-torch.abs(t_norm-bi))
            weights = ps*bilinear_weights
            vb = events_to_image_torch(xs, ys,
                    weights, device, sensor_size=sensor_size,
                    clip_out_of_range=False, interpolation=None, padding=True, default=1e-9)
        bins.append(vb)
    return bins

def events_to_neg_pos_n_voxel_torch(xs, ys, ts, ps, B, device=None,
        sensor_size=(h, w), temporal_bilinear=True):
    """
    Turn set of events to a voxel grid tensor, using temporal bilinear interpolation.
    Positive and negative events are put into separate voxel grids
    @param xs List of event x coordinates (torch tensor)
    @param ys List of event y coordinates (torch tensor)
    @param ts List of event timestamps (torch tensor)
    @param ps List of event polarities (torch tensor)
    @param B Number of bins in output voxel grids (int)
    @param device Device to put voxel grid. If left empty, same device as events
    @param sensor_size The size of the event sensor/output voxels
    @param temporal_bilinear Whether the events should be naively
        accumulated to the voxels (faster), or properly
        temporally distributed
    @returns Two voxel grids, one for positive one for negative events
    """
    zero_v = torch.tensor([0.])
    ones_v = torch.tensor([1.])
    pos_weights = torch.where(ps>0, ones_v, zero_v)
    neg_weights = torch.where(ps<=0, ones_v, zero_v)
    pos_neg_img = np.zeros((9,2,h,w))
    pos_neg_img_2d = np.zeros((9,h,w))

    voxel_pos = events_to_n_voxel_torch(xs, ys, ts, pos_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)
    voxel_neg = events_to_n_voxel_torch(xs, ys, ts, neg_weights, B, device=device,
            sensor_size=sensor_size, temporal_bilinear=temporal_bilinear)

    for i,pos in enumerate(voxel_pos):
        pos_neg_img[i,0,:,:] = pos
        pos_neg_img_2d[i,:,:] = pos
    for i,neg in enumerate(voxel_neg):
        pos_neg_img[i,1,:,:] = neg
    return pos_neg_img, pos_neg_img_2d