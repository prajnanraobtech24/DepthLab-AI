"""Read-only inventory of the downloaded NYU/KITTI/Middlebury dataset folders."""
from pathlib import Path
import argparse, json

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/downloads'); p.add_argument('--max-files',type=int,default=30); a=p.parse_args()
    root=Path(a.root)
    if not root.exists(): raise SystemExit(f'Download folder not found: {root.resolve()}')
    report={}
    for ds in ('nyu','kitti','middlebury'):
        base=root/ds; entries=[]
        if base.exists():
            for x in sorted(base.rglob('*')):
                if x.is_file(): entries.append({'path':str(x.relative_to(base)),'size_mb':round(x.stat().st_size/1024**2,2)})
        report[ds]={'exists':base.exists(),'files':len(entries),'examples':entries[:a.max_files]}
        print(f'\n[{ds.upper()}] {len(entries)} files')
        for e in entries[:a.max_files]: print(f"  {e['path']} ({e['size_mb']} MB)")
    Path('results').mkdir(exist_ok=True)
    Path('results/dataset_inventory.json').write_text(json.dumps(report,indent=2))
    print('\nSaved results/dataset_inventory.json')
if __name__=='__main__': main()
