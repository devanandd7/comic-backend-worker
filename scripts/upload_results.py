import cloudinary
import cloudinary.uploader
import os
import sys

def main():
    job_id = os.environ.get("JOB_ID")
    cloudinary_url = os.environ.get("CLOUDINARY_URL")
    
    if not job_id or not cloudinary_url:
        print("Missing JOB_ID or CLOUDINARY_URL. Skipping upload.")
        return

    # cloudinary_url format: cloudinary://API_KEY:API_SECRET@CLOUD_NAME
    # The SDK handles this automatically if set in env
    
    video_path = "output/story.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        sys.exit(1)

    print(f"Uploading {video_path} to Cloudinary with ID: {job_id}...")
    
    try:
        response = cloudinary.uploader.upload(
            video_path,
            public_id=f"comic_story_{job_id}",
            resource_type="video"
        )
        print(f"Successfully uploaded: {response['secure_url']}")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
