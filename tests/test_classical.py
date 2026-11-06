import numpy as np
from src.classical import resize_depth, guided_filter, joint_bilateral, normalize_fixed, depth_discontinuity, masked_resize, knn_fill, fill_holes
from src.metrics import compute_metrics, boundary_f1

def test_resize_scales():
    d=np.ones((8,10),np.float32)
    for scale in (2,4,8): assert resize_depth(d,scale,'nearest').shape==(8*scale,10*scale)

def test_guided_shape():
    g=np.zeros((16,16,3),np.uint8); d=np.ones((16,16),np.float32)*.5
    assert guided_filter(g,d).shape==d.shape

def test_joint_bilateral_finite_and_edge_aware():
    g=np.zeros((24,24,3),np.uint8); g[:,12:]=255
    d=np.zeros((24,24),np.float32); d[:,12:]=1
    o=joint_bilateral(g,d,diameter=5)
    assert o.shape==d.shape and np.isfinite(o).all() and o[:,5].mean()<o[:,18].mean()

def test_fixed_normalization_and_discontinuity():
    d=np.array([[1.,1.,8.],[1.,1.,8.]],np.float32); n,v=normalize_fixed(d,.1,10.)
    edge,_=depth_discontinuity(n,v); assert v.all() and edge.any()

def test_masked_resize_does_not_blend_invalid_zero():
    d=np.array([[.8,0.],[.8,0.]],np.float32); m=d>0
    out,valid=masked_resize(d,m,(4,4),'bilinear')
    assert np.allclose(out[valid],.8) and not valid[:, -1].all()

def test_knn_preserves_observations_and_fills():
    rgb=np.zeros((8,8,3),np.uint8); d=np.zeros((8,8),np.float32); m=np.zeros((8,8),bool)
    d[1,1]=.2; d[6,6]=.8; m[1,1]=m[6,6]=True
    out=knn_fill(rgb,d,m,k=2)
    assert np.isfinite(out).all() and out[1,1]==.2 and out[6,6]==.8 and out[3,3]>0

def test_fill_holes_preserves_known():
    d=np.array([[.2,0],[0,.8]],np.float32); m=d>0; out=fill_holes(d,m)
    assert out[0,0]==.2 and out[1,1]==.8 and np.all(out>0)

def test_metrics_physical_units_and_min_depth_validity():
    gt=np.array([[.1,1.0],[2.0,3.0]],np.float32); pred=gt.copy(); valid=np.ones_like(gt,bool)
    met=compute_metrics(pred,gt,valid,.1,10)
    assert met['MAE_m']==0 and met['RMSE_m']==0 and met['valid_pixels']==4
    assert boundary_f1(gt,gt,valid)==1

def test_metrics_reports_prediction_coverage():
    gt=np.ones((16,16),np.float32)*2; pred=gt.copy(); pred[:,:8]=0
    met=compute_metrics(pred,gt,min_depth=.1,max_depth=10.)
    assert met['prediction_coverage']==.5
