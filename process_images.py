"""
Script to process all images in the input folder and create headshots.
"""
from pathlib import Path
from profile_photo import create_headshot

# Input and output directories
input_dir = Path('profile_photo/input/drive-download-20260102T074514Z-1-001')
output_dir = Path('profile_photo/output/headshots')

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Get all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
image_files = [f for f in input_dir.iterdir() if f.suffix in image_extensions]

print(f"Found {len(image_files)} images to process")
print(f"Output directory: {output_dir}")

# Process each image
successful = 0
failed = 0

for i, image_path in enumerate(image_files, 1):
    print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
    
    try:
        # Create headshot
        photo = create_headshot(
            image_path,
            output_dir=output_dir,
            debug=False
        )
        
        # Save the result
        photo.save_image(output_dir)
        print(f"  [OK] Successfully created headshot")
        successful += 1
        
    except Exception as e:
        print(f"  [ERROR] Error processing {image_path.name}: {str(e)}")
        failed += 1
        continue

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Total: {len(image_files)}")
print(f"Output saved to: {output_dir}")

