"""
Script to process all images in the input folder and create headshots with background removal.
"""
from pathlib import Path
from profile_photo import create_headshot

# Input and output directories
input_dir = Path('profile_photo/input/drive-download-20260102T074514Z-1-001')
output_dir = Path('profile_photo/output/headshots-bg-removed')

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Get all image files
image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
image_files = [f for f in input_dir.iterdir() if f.suffix in image_extensions]

print(f"Found {len(image_files)} images to process")
print(f"Output directory: {output_dir}")
print(f"Background removal: ENABLED (transparent PNG)")

# Process each image
successful = 0
failed = 0
skipped = 0

for i, image_path in enumerate(image_files, 1):
    print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
    
    # Check if output already exists
    filename, ext = image_path.stem, image_path.suffix
    output_filename = f"{filename}-out.png"  # PNG for transparency
    output_path = output_dir / output_filename
    
    if output_path.exists():
        print(f"  [SKIP] Output already exists: {output_filename}")
        skipped += 1
        continue
    
    try:
        # Create headshot with background removal
        photo = create_headshot(
            image_path,
            output_dir=output_dir,
            remove_bg=True,  # Enable background removal
            debug=False
        )
        
        # Save the result (will be PNG with transparency)
        photo.save_image(output_dir)
        print(f"  [OK] Successfully created headshot with background removed")
        successful += 1
        
    except Exception as e:
        print(f"  [ERROR] Error processing {image_path.name}: {str(e)}")
        failed += 1
        continue

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Skipped (already exists): {skipped}")
print(f"  Total: {len(image_files)}")
print(f"Output saved to: {output_dir}")

