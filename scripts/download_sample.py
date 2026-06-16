import subprocess
import os

print("Downloading full sample video without sections...")
cmd = [
    "yt-dlp",
    "ytsearch1:dog crossing street dashcam short",
    "-f", "mp4",
    "-o", "/workspace/demo_input/github_sample.mp4",
    "--force-overwrites"
]
subprocess.run(cmd, check=True)
print("Done!")
