import numpy as np
from PIL import Image
from prepare_kitti import read_kitti_depth, split_for_key
from prepare_middlebury import read_pfm

def test_kitti_depth_png_decodes_256_units_per_metre(tmp_path):
    p=tmp_path/'depth.png'; Image.fromarray(np.array([[0,256,512]],dtype=np.uint16)).save(p)
    d=read_kitti_depth(p)
    assert d.tolist()==[[0.0,1.0,2.0]]

def test_kitti_split_detection():
    assert split_for_key('val_selection_cropped/0001.png')=='val'
    assert split_for_key('train/2011_drive/0001.png')=='train'
    assert split_for_key('unknown/0001.png')=='unassigned'

def test_pfm_reader(tmp_path):
    p=tmp_path/'disp.pfm'
    arr=np.array([[1.0,2.0],[3.0,4.0]],dtype='<f4')
    with open(p,'wb') as f:
        f.write(b'Pf\n2 2\n-1.0\n'); f.write(np.flipud(arr).tobytes())
    got,scale=read_pfm(p)
    assert scale==1.0 and np.allclose(got,arr)
