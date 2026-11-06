import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from server import app

def test_home_and_health():
    c=TestClient(app); assert c.get('/').status_code==200; assert c.get('/api/health').json()['status']=='ok'

def test_reconstruction_api_with_gt():
    c=TestClient(app)
    rgb=np.zeros((64,80,3),np.uint8); rgb[:,:40]=[180,50,40]; rgb[:,40:]=[20,160,180]
    depth=np.ones((16,20),np.float32)*2; depth[:,10:]=5
    gt=np.ones((64,80),np.float32)*2; gt[:,40:]=5
    b=io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG')
    d=io.BytesIO(); np.save(d,depth)
    g=io.BytesIO(); np.save(g,gt)
    r=c.post('/api/reconstruct',data={'scale':'4','method':'joint_bilateral','interpolation':'bilinear','mode':'Classical','alpha':'0.5','depth_profile':'NYU indoor · 10 m','depth_units':'metres','depth_input_type':'auto'},files={'rgb_file':('rgb.png',b.getvalue(),'image/png'),'depth_file':('depth.npy',d.getvalue(),'application/octet-stream'),'gt_file':('gt.npy',g.getvalue(),'application/octet-stream')})
    assert r.status_code==200, r.text
    payload=r.json(); assert payload['scale']==4 and payload['metrics']['RMSE_m'] is not None
    assert payload['output'].startswith('data:image/png;base64,') and payload['knn'].startswith('data:image/png;base64,')

def test_full_resolution_sparse_is_not_downsampled():
    c=TestClient(app); rgb=np.zeros((64,64,3),np.uint8); depth=np.zeros((64,64),np.float32); depth[::4,::4]=2
    b=io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG'); d=io.BytesIO(); np.save(d,depth)
    r=c.post('/api/reconstruct',data={'scale':'4','mode':'KNN','depth_profile':'NYU indoor · 10 m','depth_units':'metres','depth_input_type':'auto'},files={'rgb_file':('rgb.png',b.getvalue(),'image/png'),'depth_file':('sparse.npy',d.getvalue(),'application/octet-stream')})
    assert r.status_code==200, r.text
    assert r.json()['input_kind']=='full_resolution_sparse'

def test_bad_scale_ratio_is_rejected():
    c=TestClient(app); rgb=np.zeros((64,64,3),np.uint8); depth=np.ones((8,8),np.float32)*2
    b=io.BytesIO(); Image.fromarray(rgb).save(b,format='PNG'); d=io.BytesIO(); np.save(d,depth)
    r=c.post('/api/reconstruct',data={'scale':'4','mode':'KNN','depth_profile':'NYU indoor · 10 m','depth_units':'metres'},files={'rgb_file':('rgb.png',b.getvalue(),'image/png'),'depth_file':('depth.npy',d.getvalue(),'application/octet-stream')})
    assert r.status_code==400 and 'do not match' in r.json()['detail']

def test_png_export_encoding_supports_kitti_range():
    from server import png_depth
    raw=png_depth(np.array([[0.0,70.0]],np.float32),256)
    img=np.asarray(Image.open(io.BytesIO(raw)))
    assert img.dtype==np.uint16 and img[0,0]==0 and abs(float(img[0,1])/256.0-70.0)<.01
