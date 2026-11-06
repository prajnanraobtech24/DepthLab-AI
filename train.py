"""Train compact RGB-guided U-Net using validity-aware synthetic 2x/4x/8x inputs."""
import argparse, glob, os, json, random
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from src.model import DepthRefineNet

class DensePairs(Dataset):
    def __init__(self,root,split,height=240,width=320,augment=False,eval_scale=4):
        self.root=root; self.split=split; self.height=height; self.width=width; self.augment=augment; self.eval_scale=eval_scale
        rg={os.path.basename(x):x for x in glob.glob(f'{root}/{split}/rgb/*.npy')}
        gt={os.path.basename(x):x for x in glob.glob(f'{root}/{split}/gt/*.npy')}
        self.pairs=[(rg[k],gt[k]) for k in sorted(set(rg)&set(gt))]
        if not self.pairs: raise RuntimeError(f'No matching RGB/GT NPY pairs in {root}/{split}')
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        rp,gp=self.pairs[i]
        rgb=np.load(rp).astype(np.float32); rgb=rgb/255.0 if rgb.max()>1 else rgb
        gt=np.load(gp).astype(np.float32)
        # Resize GT validity-aware; do not interpolate invalid zero values into valid depths.
        valid=np.isfinite(gt)&(gt>=0.1)&(gt<=10.0)&(gt>0)
        num=cv2.resize(np.where(valid,gt,0),(self.width,self.height),interpolation=cv2.INTER_AREA)
        den=cv2.resize(valid.astype(np.float32),(self.width,self.height),interpolation=cv2.INTER_AREA)
        gt=num/np.maximum(den,1e-6); valid=den>=0.75
        gt_norm=np.zeros_like(gt,np.float32); gt_norm[valid]=(gt[valid]-.1)/9.9
        rgb=cv2.resize(rgb,(self.width,self.height),interpolation=cv2.INTER_AREA)
        scale=int(random.choice([2,4,8])) if self.augment else self.eval_scale
        lh,lw=self.height//scale,self.width//scale
        # Validity-weighted area downsampling: no invalid zero contamination.
        num=cv2.resize(gt_norm*valid.astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
        den=cv2.resize(valid.astype(np.float32),(lw,lh),interpolation=cv2.INTER_AREA)
        low=num/np.maximum(den,1e-6); coarse_valid=den>=0.75
        if self.augment:
            coarse_valid &= np.random.rand(lh,lw)>=np.random.uniform(0.0,0.25)
            low += np.random.normal(0,np.random.uniform(0,.005),low.shape).astype(np.float32)
        low=np.clip(low,0,1); low[~coarse_valid]=0
        # Nearest-neighbour fill for input channel; mask separately tells the net what was observed.
        from scipy.ndimage import distance_transform_edt
        if coarse_valid.any():
            _,ind=distance_transform_edt(~coarse_valid,return_indices=True)
            low_filled=low.copy(); low_filled[~coarse_valid]=low[ind[0][~coarse_valid],ind[1][~coarse_valid]]
        else: low_filled=np.zeros_like(low)
        up=cv2.resize(low_filled,(self.width,self.height),interpolation=cv2.INTER_NEAREST)
        mask=cv2.resize(coarse_valid.astype(np.float32),(self.width,self.height),interpolation=cv2.INTER_NEAREST)
        if self.augment and random.random()<.5:
            rgb,gt_norm,valid,up,mask=[np.ascontiguousarray(x[:,::-1]) for x in (rgb,gt_norm,valid,up,mask)]
        x=np.concatenate([rgb,up[...,None],mask[...,None]],axis=2).transpose(2,0,1)
        return torch.from_numpy(x.astype(np.float32)),torch.from_numpy(gt_norm[None].astype(np.float32)),torch.from_numpy(valid[None].astype(np.float32))

def gradient_loss(pred,target,mask):
    px=pred[:,:,:,1:]-pred[:,:,:,:-1]; tx=target[:,:,:,1:]-target[:,:,:,:-1]
    py=pred[:,:,1:,:]-pred[:,:,:-1,:]; ty=target[:,:,1:,:]-target[:,:,:-1,:]
    mx=mask[:,:,:,1:]*mask[:,:,:,:-1]; my=mask[:,:,1:,:]*mask[:,:,:-1,:]
    return ((px-tx).abs()*mx).sum()/(mx.sum()+1e-6)+((py-ty).abs()*my).sum()/(my.sum()+1e-6)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/nyu'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--out',default='checkpoints/depthrefinenet.pt'); p.add_argument('--height',type=int,default=240); p.add_argument('--width',type=int,default=320); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
    if a.height%4 or a.width%4: raise SystemExit('Height and width must be divisible by 4')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    tr=DensePairs(a.data,'train',a.height,a.width,True)
    vals=[DensePairs(a.data,'val',a.height,a.width,False,s) for s in (2,4,8)]
    train=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0)
    vloads=[DataLoader(v,batch_size=a.batch_size,shuffle=False,num_workers=0) for v in vals]
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model=DepthRefineNet().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); best=float('inf'); history=[]
    os.makedirs(os.path.dirname(a.out) or '.',exist_ok=True)
    for ep in range(a.epochs):
        model.train(); losses=[]
        for x,y,m in train:
            x,y,m=x.to(dev),y.to(dev),m.to(dev); pred=model(x)
            l1=((pred-y).abs()*m).sum()/(m.sum()+1e-6); loss=l1+.1*gradient_loss(pred,y,m)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.item()))
        model.eval(); by_scale={}
        with torch.inference_mode():
            for sc,loader in zip((2,4,8),vloads):
                scores=[]
                for x,y,m in loader:
                    x,y,m=x.to(dev),y.to(dev),m.to(dev); pred=model(x)
                    scores.append(float(((pred-y).abs()*m).sum()/(m.sum()+1e-6)))
                by_scale[str(sc)]=float(np.mean(scores))
        val=float(np.mean(list(by_scale.values()))); row={'epoch':ep+1,'train_loss':float(np.mean(losses)),'val_mae_normalized_by_scale':by_scale,'val_mean':val}; history.append(row)
        print(json.dumps(row))
        if val<best:
            best=val; torch.save({'model_state_dict':model.state_dict(),'depth_min_m':.1,'depth_max_m':10.,'input_channels':5,'training_scales':[2,4,8],'architecture':'DepthRefineNet-U-Net','best_val_mae_normalized':best},a.out)
    out_dir=os.path.dirname(a.out) or '.'
    with open(os.path.join(out_dir,'training_history.json'),'w') as f: json.dump(history,f,indent=2)
    print(f'Best validation normalized MAE={best:.6f}; checkpoint={a.out}; device={dev}')
if __name__=='__main__': main()
