"""Keep-alive script for DepthLab AI Render deployment to prevent free-tier spindown."""
import time, os, sys
import urllib.request
import json

TARGET_URL = os.getenv("DEPTHLAB_DEPLOY_URL", "https://depthlab-ai.onrender.com/ping")
INTERVAL_SECONDS = int(os.getenv("PING_INTERVAL", "600"))  # Ping every 10 minutes

def ping_server():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pinging {TARGET_URL}...")
    try:
        req = urllib.request.Request(
            TARGET_URL,
            headers={"User-Agent": "DepthLab-KeepAlive-Monitor/2.0"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] HTTP {status}: {body.strip()}")
            return True
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Ping failed: {e}")
        return False

def main():
    print(f"Starting DepthLab AI Keep-Alive Daemon for: {TARGET_URL}")
    print(f"Interval: {INTERVAL_SECONDS} seconds (10 minutes)")
    while True:
        ping_server()
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        ping_server()
    else:
        main()
