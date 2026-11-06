"""Execute all 50 gallery benchmark scenes through the actual FastAPI /api/reconstruct endpoint."""
import json, time
from pathlib import Path
from fastapi.testclient import TestClient
from server import app

def main():
    client = TestClient(app)
    manifest_p = Path("web/presets/gallery_manifest.json")
    with open(manifest_p, "r") as f:
        manifest = json.load(f)
    
    print(f"Loaded {len(manifest)} scenes from manifest. Starting live /api/reconstruct benchmarks...")
    results = []
    
    total_start = time.perf_counter()
    for idx, scene in enumerate(manifest):
        sid = scene["id"]
        title = scene["title"]
        rgb_path = Path("web") / scene["rgbUrl"].replace("/assets/", "")
        depth_path = Path("web") / scene["depthUrl"].replace("/assets/", "")
        gt_path = Path("web") / scene["gtUrl"].replace("/assets/", "")
        
        with open(rgb_path, "rb") as fr, open(depth_path, "rb") as fd, open(gt_path, "rb") as fg:
            files = {
                "rgb_file": (f"{sid}_rgb.png", fr.read(), "image/png"),
                "depth_file": (f"{sid}_depth.png", fd.read(), "image/png"),
                "gt_file": (f"{sid}_gt.png", fg.read(), "image/png")
            }
            data = {
                "scale": str(scene["scale"]),
                "method": scene["method"],
                "interpolation": "bilinear",
                "mode": scene["mode"],
                "alpha": "0.5",
                "depth_profile": scene["profile"],
                "depth_units": scene["units"],
                "depth_input_type": scene["inputType"],
                "colormap": "turbo"
            }
            
            t0 = time.perf_counter()
            resp = client.post("/api/reconstruct", data=data, files=files)
            dur = (time.perf_counter() - t0) * 1000
            
            assert resp.status_code == 200, f"Scene {sid} failed: {resp.text}"
            res_json = resp.json()
            metrics = res_json.get("metrics", {})
            
            item_res = {
                "index": idx + 1,
                "id": sid,
                "title": title,
                "dataset": scene["dataset"],
                "category": scene["category"],
                "latency_ms": round(res_json["latency_ms"], 1),
                "total_roundtrip_ms": round(dur, 1),
                "RMSE_m": metrics.get("RMSE_m"),
                "MAE_m": metrics.get("MAE_m"),
                "Boundary_F1_2px": metrics.get("Boundary_F1_2px"),
                "PSNR_dB": metrics.get("PSNR_dB"),
                "SSIM": metrics.get("SSIM"),
                "valid_fraction": res_json["valid_fraction"],
                "status": "PASS"
            }
            results.append(item_res)
            rmse_str = f"{metrics.get('RMSE_m', 0.0):.4f}m" if metrics.get('RMSE_m') else "N/A"
            f1_str = f"{metrics.get('Boundary_F1_2px', 0.0):.3f}" if metrics.get('Boundary_F1_2px') else "N/A"
            print(f"[{idx+1:02d}/50] {sid:8s} | {scene['dataset']:16s} | {title:28s} | Latency: {dur:5.0f}ms | RMSE: {rmse_str} | Boundary F1: {f1_str} -> PASS")

    total_dur = time.perf_counter() - total_start
    out_file = Path("results/gallery_benchmarks_50.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"summary": {"total_scenes": len(results), "passed": len(results), "total_duration_sec": round(total_dur, 2)}, "scenes": results}, f, indent=2)
    
    print(f"\nALL 50/50 SCENES SUCCESSFULLY EXECUTED AND BENCHMARKED IN {total_dur:.1f}s!")
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
