import subprocess
import os

def frames_to_video(input_pattern, output_path, framerate=1):
    try:
        # Correct path to ffmpeg.exe
        ffmpeg_path = os.path.abspath(os.path.join("ffmpeg", "bin", "ffmpeg.exe"))
        print("🔍 Using FFmpeg path:", ffmpeg_path)

        if not os.path.isfile(ffmpeg_path):
            print("❌ ffmpeg.exe not found at:", ffmpeg_path)
            return

        command = [
            ffmpeg_path,
            "-framerate", str(framerate),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        subprocess.run(command, check=True)
        print("✅ Video successfully created at:", output_path)

    except subprocess.CalledProcessError as e:
        print("❌ Error during FFmpeg execution:", e)

if __name__ == "__main__":
    input_pattern = "segmented_frames/frame_%04d.jpg"
    output_path = "output_video.mp4"

    frames_to_video(input_pattern, output_path)
