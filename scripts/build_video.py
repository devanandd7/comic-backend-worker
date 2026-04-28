import os
import glob
import subprocess

def main():
    final_frames = sorted(glob.glob("output/panel_*.jpg"))
    if not final_frames:
        print("No panel_*.jpg frames found in output/. Skipping video build.")
        return

    print(f"Found {len(final_frames)} frames. Building video...")

    # Write sorted frame list for FFmpeg concat
    with open("output/frames.txt", "w") as f:
        for frame in final_frames:
            f.write(f"file '../{frame}'\n")
            f.write("duration 2\n")   # 2 seconds per panel

    # Build video using concat demuxer (most reliable across platforms)
    # Scale to 1080x1080 square, force even dimensions, H.264 baseline
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "output/frames.txt",
        "-vf", "scale=1080:1080:force_original_aspect_ratio=decrease,pad=1080:1080:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "output/story.mp4"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✓ Successfully generated output/story.mp4")
    else:
        print("✗ FFmpeg failed:")
        print(result.stderr[-2000:])   # Print last 2000 chars of error

if __name__ == "__main__":
    main()
