import cv2
import numpy as np
import os
import sys
from pathlib import Path
import math

# Add parent directory to Python path to find config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
sys.path.insert(0, project_root)


class PhotoCollage:
    def __init__(self, output_dir="output", collage_output="output/photo_collage.jpg"):
        """
        Initialize the PhotoCollage class.
        
        Args:
            output_dir (str): Directory containing filtered photos
            collage_output (str): Output path for the final collage
        """
        self.output_dir = output_dir
        self.collage_output = collage_output
        self.grid_size = 4  # 4x4 grid
        self.max_photos = self.grid_size * self.grid_size  # 16 photos max
        
    def get_filtered_photos(self):
        """
        Get all filtered photos from the output directory (excluding the original input).
        
        Returns:
            list: List of image file paths
        """
        if not os.path.exists(self.output_dir):
            print(f"❌ Output directory not found: {self.output_dir}")
            return []
        
        # Get all image files from output directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        photo_files = []
        
        for file in os.listdir(self.output_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Skip the collage files to avoid recursion
                if 'collage' in file.lower():
                    continue
                    
                # Skip obviously large original files (typically much larger than filtered versions)
                file_path = os.path.join(self.output_dir, file)
                if os.path.isfile(file_path):
                    photo_files.append(file_path)
        
        # Sort files for consistent ordering
        photo_files.sort()
        
        print(f"Found {len(photo_files)} filtered photos:")
        for i, photo in enumerate(photo_files, 1):
            print(f"  {i}. {os.path.basename(photo)}")
            
        return photo_files
    
    def resize_image_to_fit(self, image, target_width, target_height):
        """
        Resize image to fit within target dimensions while maintaining aspect ratio.
        
        Args:
            image: OpenCV image array
            target_width (int): Target width
            target_height (int): Target height
            
        Returns:
            numpy.ndarray: Resized image
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factors
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)  # Use smaller scale to fit within bounds
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create a canvas with target dimensions and center the image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def add_photo_label(self, image, label, font_scale=0.4, thickness=1):
        """
        Add a label to the bottom of the image.
        
        Args:
            image: OpenCV image array
            label (str): Label text
            font_scale (float): Font scale
            thickness (int): Text thickness
            
        Returns:
            numpy.ndarray: Image with label
        """
        labeled_image = image.copy()
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Create a semi-transparent overlay for better readability
        overlay = labeled_image.copy()
        h, w = labeled_image.shape[:2]
        
        # Draw rectangle for text background
        rect_height = text_height + baseline + 10
        cv2.rectangle(overlay, (0, h - rect_height), (w, h), (0, 0, 0), -1)
        
        # Blend overlay with original image
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, labeled_image, 1 - alpha, 0, labeled_image)
        
        # Add text
        text_x = (w - text_width) // 2
        text_y = h - baseline - 5
        cv2.putText(labeled_image, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return labeled_image
    
    def create_collage(self, cell_width=400, cell_height=300, spacing=10, background_color=(20, 20, 20)):
        """
        Create a collage from filtered photos arranged in a grid.
        
        Args:
            cell_width (int): Width of each photo cell
            cell_height (int): Height of each photo cell
            spacing (int): Spacing between photos
            background_color (tuple): RGB background color
            
        Returns:
            bool: True if collage was created successfully, False otherwise
        """
        photo_files = self.get_filtered_photos()
        
        if not photo_files:
            print("❌ No photos found to create collage")
            return False
        
        # Limit to maximum photos that fit in grid
        if len(photo_files) > self.max_photos:
            print(f"⚠️  Found {len(photo_files)} photos, using first {self.max_photos} for {self.grid_size}x{self.grid_size} grid")
            photo_files = photo_files[:self.max_photos]
        
        # Calculate actual grid dimensions based on number of photos
        num_photos = len(photo_files)
        cols = self.grid_size
        rows = math.ceil(num_photos / cols)
        
        print(f"Creating {rows}x{cols} collage with {num_photos} photos...")
        
        # Calculate canvas dimensions
        canvas_width = cols * cell_width + (cols + 1) * spacing
        canvas_height = rows * cell_height + (rows + 1) * spacing
        
        # Create canvas
        canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)
        
        # Process each photo
        for idx, photo_path in enumerate(photo_files):
            try:
                # Load image
                img = cv2.imread(photo_path)
                if img is None:
                    print(f"⚠️  Could not load {os.path.basename(photo_path)}, skipping...")
                    continue
                
                # Resize image to fit cell
                resized_img = self.resize_image_to_fit(img, cell_width, cell_height)
                
                # Add label with filter name
                filter_name = os.path.splitext(os.path.basename(photo_path))[0].replace('_', ' ').title()
                labeled_img = self.add_photo_label(resized_img, filter_name)
                
                # Calculate position in grid
                row = idx // cols
                col = idx % cols
                
                # Calculate pixel position
                y = row * (cell_height + spacing) + spacing
                x = col * (cell_width + spacing) + spacing
                
                # Place image on canvas
                canvas[y:y + cell_height, x:x + cell_width] = labeled_img
                
                print(f"  ✅ Placed {filter_name} at position ({row + 1}, {col + 1})")
                
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(photo_path)}: {e}")
                continue
        
        # Save collage
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(self.collage_output), exist_ok=True)
            
            success = cv2.imwrite(self.collage_output, canvas)
            if success:
                print(f"\n🎉 Collage created successfully!")
                print(f"📁 Saved to: {self.collage_output}")
                print(f"📐 Dimensions: {canvas_width}x{canvas_height} pixels")
                return True
            else:
                print(f"❌ Failed to save collage to {self.collage_output}")
                return False
                
        except Exception as e:
            print(f"❌ Error saving collage: {e}")
            return False


def main():
    """Main function to run the collage creation."""
    print("🖼️  Film Filter Photo Collage Creator")
    print("=" * 50)
    
    # Create collage instance
    collage_creator = PhotoCollage()
    
    # Create the collage
    success = collage_creator.create_collage()
    
    if success:
        print("\n✨ Collage creation completed successfully!")
    else:
        print("\n❌ Collage creation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()