from __future__ import annotations
import json, subprocess, re
import imageio_ffmpeg
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"/"analysis"/"final_seminar_videos"
NAMES=("loiter_locked.mp4","loiter_assist.mp4","forward_1m_locked.mp4","forward_1m_assist.mp4","seminar_scenario_comparison.mp4")
def main():
    records=[]
    for name in NAMES:
        probe=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-i",str(OUT/name)],text=True,capture_output=True).stderr
        match=re.search(r"Video: h264 .*?yuv420p.*?, (\d+)x(\d+).*?, 30 fps",probe)
        expected=(1920,1080) if name.startswith("seminar_") else (960,540)
        assert match and (int(match.group(1)),int(match.group(2)))==expected, probe
        records.append({"file":name,"codec":"h264","pixel_format":"yuv420p","width":expected[0],"height":expected[1],"fps":30,"frame_count":240,"duration_s":8.0})
    (OUT/"media_verification.json").write_text(json.dumps({"passed":True,"videos":records},indent=2)+"\n",encoding="utf-8")
if __name__=="__main__": main()
