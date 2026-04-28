import os
import glob

def main():
    # Make sure we have final frames
    final_frames = glob.glob("output/panel_*.jpg")
    if not final_frames:
        print("No panel_*.jpg frames found in output/. Skipping video build.")
        return
    
    print(f"Found {len(final_frames)} frames. Building video...")
    
    cmd = """
    ffmpeg -y -framerate 1 -pattern_type glob -i 'output/panel_*.jpg' -c:v libx264 -pix_fmt yuv420p output/story.mp4
    """
    
    exit_code = os.system(cmd)
    
    if exit_code == 0:
        print("Successfully generated output/story.mp4")
    else:
        print("Failed to generate video.")

if __name__ == "__main__":
    main()
