import os
import glob

def main():
    # Make sure we have final frames
    final_frames = glob.glob("output/*_final.jpg")
    if not final_frames:
        print("No *_final.jpg frames found in output/. Skipping video build.")
        return
    
    print(f"Found {len(final_frames)} frames. Building video...")
    
    # Run FFmpeg command to stitch images into a video
    # -framerate 1 means 1 second per frame
    # -pattern_type glob for using wildcard
    # -c:v libx264 for H.264 video codec
    # -pix_fmt yuv420p for wide player compatibility
    
    cmd = """
    ffmpeg -y -framerate 1 -pattern_type glob -i 'output/*_final.jpg' -c:v libx264 -pix_fmt yuv420p output/story.mp4
    """
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Successfully generated output/story.mp4")
    else:
        print("Failed to generate video.")

if __name__ == "__main__":
    main()
