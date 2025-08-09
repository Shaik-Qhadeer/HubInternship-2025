import os
import subprocess

def extract_frames(video_path, output_dir, interval=3):
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps=1/{interval}',
        os.path.join(frames_dir, 'frame_%04d.jpg')
    ]
    subprocess.run(cmd, check=True)
    print(f"Frames saved to: {frames_dir}")

if __name__ == "__main__":
    video_path = "videoplayback (1).mp4"  # Your video
    output_dir = "frames/images"     # Where extracted frames will go
    extract_frames(video_path, output_dir, interval=3)
