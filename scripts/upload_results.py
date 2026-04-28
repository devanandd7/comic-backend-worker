import cloudinary
import cloudinary.uploader
import os
import sys
import glob

def main():
    job_id = os.environ.get("JOB_ID")
    cloudinary_url = os.environ.get("CLOUDINARY_URL")
    
    if not job_id or not cloudinary_url:
        print("Missing JOB_ID or CLOUDINARY_URL. Skipping upload.")
        return

    # Upload Debug Grid
    debug_path = "output/debug_grid.jpg"
    if os.path.exists(debug_path):
        print(f"Uploading debug grid...")
        cloudinary.uploader.upload(
            debug_path,
            public_id=f"comic_results/{job_id}/debug_grid",
            resource_type="image"
        )

    # Upload all panels
    panels = sorted(glob.glob("output/panel_*.jpg"))
    print(f"Uploading {len(panels)} panels...")
    for i, panel_path in enumerate(panels):
        cloudinary.uploader.upload(
            panel_path,
            public_id=f"comic_results/{job_id}/panel_{i+1:03d}",
            resource_type="image"
        )

    # Upload Final Video
    video_path = "output/story.mp4"
    if os.path.exists(video_path):
        print(f"Uploading video...")
        response = cloudinary.uploader.upload(
            video_path,
            public_id=f"comic_results/{job_id}/story",
            resource_type="video"
        )
        print(f"Final Video URL: {response['secure_url']}")
    
    print(f"Done! Results available under comic_results/{job_id}/")

if __name__ == "__main__":
    main()
