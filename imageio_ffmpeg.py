from imageio_ffmpeg import get_ffmpeg_exe
import subprocess
import sys

# Get the path to the FFmpeg executable
ffmpeg_path = r"imageio_ffmpeg\binaries\ffmpeg-win64-v4.2.2.exe"

# Get input and output video paths from command-line arguments
input_video_path = sys.argv[1]
output_video_path = sys.argv[2]

# FFmpeg command
command = [
    ffmpeg_path, '-i', input_video_path,
    '-vcodec', 'libx264', '-acodec', 'aac',
    '-strict', 'experimental', output_video_path
]

# Execute the FFmpeg command
subprocess.run(command)
