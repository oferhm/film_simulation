"""
Demo script to test film simulation filters
"""

import cv2
import numpy as np
from film_filters import (
    kodak_portra, 
    fuji_film, 
    disposable_camera, 
    cinematic_film, 
    vintage_film,
    FilmSimulator,
    apply_all_filters
)
from config import input


def create_test_image():
    """Create a test image with color gradients for testing"""
    height, width = 600, 800
    
    # Create RGB gradient image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create color bars
    bar_width = width // 6
    
    # Red
    img[:, 0:bar_width] = [0, 0, 255]
    # Orange
    img[:, bar_width:bar_width*2] = [0, 165, 255]
    # Yellow
    img[:, bar_width*2:bar_width*3] = [0, 255, 255]
    # Green
    img[:, bar_width*3:bar_width*4] = [0, 255, 0]
    # Blue
    img[:, bar_width*4:bar_width*5] = [255, 0, 0]
    # Purple
    img[:, bar_width*5:] = [255, 0, 255]
    
    # Add brightness gradient
    for y in range(height):
        brightness = y / height
        img[y, :] = (img[y, :] * brightness).astype(np.uint8)
    
    return img


def demo_single_filter():
    """Demo: Apply single filter to an image"""
    print("\n=== Demo 1: Single Filter ===")
    
    # Create test image
    # img = create_test_image()
    img = cv2.imread(input)  # Try to load real photo if exists
    if img is None:
        print(f"❌ Could not load image: {input}")
        print("Using test image instead...")
        img = create_test_image()
    cv2.imwrite("test_original.jpg", img)
    print("✓ Created test image: test_original.jpg")
    
    # Apply Kodak Portra filter
    result = kodak_portra(image=img)
    cv2.imwrite("test_portra.jpg", result)
    print("✓ Applied Kodak Portra filter: test_portra.jpg")


def demo_all_filters():
    """Demo: Apply all 5 filters"""
    print("\n=== Demo 2: All Filters ===")
    
    # Create test image
    # img = create_test_image()
    img = cv2.imread(input)  # Try to load real photo if exists
    if img is None:
        print(f"❌ Could not load image: {input}")
        print("Using test image instead...")
        img = create_test_image()
    
    filters = {
        'Kodak Portra': kodak_portra,
        'Fuji Film': fuji_film,
        'Disposable Camera': disposable_camera,
        'Cinematic': cinematic_film,
        'Vintage': vintage_film
    }
    
    for name, filter_func in filters.items():
        result = filter_func(image=img)
        filename = f"test_{name.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(filename, result)
        print(f"✓ {name}: {filename}")


def demo_custom_filter():
    """Demo: Create custom filter with FilmSimulator"""
    print("\n=== Demo 3: Custom Filter ===")
    
    img = cv2.imread(input)  # Try to load real photo if exists
    if img is None:
        print(f"❌ Could not load image: {input}")
        print("Using test image instead...")
    
    # Create custom film look
    sim = FilmSimulator(image=img)
    sim.adjust_temperature(150)  # Slight warm
    sim.adjust_saturation(-20)   # Desaturate
    sim.adjust_contrast(15)      # Increase contrast
    sim.adjust_blacks(8)         # Lift blacks (fade)
    sim.add_grain(30, 20)        # Add grain
    sim.add_vignette(25)         # Add vignette
    
    result = sim.get_result()
    cv2.imwrite("test_custom.jpg", result)
    print("✓ Custom filter: test_custom.jpg")


def demo_with_real_photo(image_path: str):
    """Demo: Apply filters to a real photo"""
    print(f"\n=== Demo 4: Real Photo ({image_path}) ===")
    
    # Check if file exists
    import os
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("   Place a photo.jpg in the same directory and try again")
        return
    
    # Apply all filters
    apply_all_filters(image_path, output_dir="filtered_photos/")


def compare_filters_side_by_side():
    """Create a comparison grid of all filters"""
    print("\n=== Demo 5: Side-by-Side Comparison ===")
    
    # load image
    img = cv2.imread(input)
    
    # Resize for grid
    scale = 0.4
    small_img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Apply filters
    filters = {
        'Original': img,
        'Portra': kodak_portra(image=img),
        'Fuji': fuji_film(image=img),
        'Disposable': disposable_camera(image=img),
        'Cinematic': cinematic_film(image=img),
        'Vintage': vintage_film(image=img)
    }
    
    # Resize all
    resized = {name: cv2.resize(img, None, fx=scale, fy=scale) 
               for name, img in filters.items()}
    
    # Add labels
    for name, img in resized.items():
        # Add white bar at bottom for text
        h, w = img.shape[:2]
        bar = np.ones((40, w, 3), dtype=np.uint8) * 255
        img_with_label = np.vstack([img, bar])
        
        # Add text
        cv2.putText(img_with_label, name, (10, h + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        resized[name] = img_with_label
    
    # Create grid (2 rows x 3 columns)
    row1 = np.hstack([resized['Original'], resized['Portra'], resized['Fuji']])
    row2 = np.hstack([resized['Disposable'], resized['Cinematic'], resized['Vintage']])
    grid = np.vstack([row1, row2])
    
    cv2.imwrite("comparison_grid.jpg", grid)
    print("✓ Created comparison grid: comparison_grid.jpg")


def show_filter_parameters():
    """Display filter parameters for reference"""
    print("\n=== Film Filter Parameters ===\n")
    
    print("1. KODAK PORTRA (Wedding/Portrait)")
    print("   • Highlights: -30")
    print("   • Shadows: +20")
    print("   • Blacks: +10 (fade)")
    print("   • Temperature: +200 (warm)")
    print("   • Saturation: -10")
    print("   • Grain: 25\n")
    
    print("2. FUJI FILM (Strong greens & blues)")
    print("   • Contrast: +10")
    print("   • Highlights: -20")
    print("   • Temperature: -150 (cool)")
    print("   • Tint: -5 (green)")
    print("   • Grain: 20\n")
    
    print("3. DISPOSABLE CAMERA (90s nostalgic)")
    print("   • Contrast: +25")
    print("   • Highlights: -40")
    print("   • Saturation: +15")
    print("   • Grain: 40 (high)")
    print("   • Vignette: 30")
    print("   • Clarity: -10\n")
    
    print("4. CINEMATIC (Movie look)")
    print("   • Split tone: Teal shadows → Orange highlights")
    print("   • Contrast: -5 (soft)")
    print("   • Clarity: -15 (glow)")
    print("   • Grain: 20\n")
    
    print("5. VINTAGE (Retro warm)")
    print("   • Blacks: +15 (fade)")
    print("   • Temperature: +300 (very warm)")
    print("   • Saturation: -15")
    print("   • Contrast: -10")
    print("   • Grain: 30\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  FILM SIMULATION FILTERS - DEMO")
    print("=" * 60)
    
    # Show parameters
    show_filter_parameters()
    
    # Run demos
    # demo_single_filter()
    demo_all_filters()
    # demo_custom_filter()
    compare_filters_side_by_side()
    
    # Try with real photo if exists
    demo_with_real_photo("photo.jpg")
    
    print("\n" + "=" * 60)
    print("  ✅ ALL DEMOS COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • test_*.jpg - Test images with different filters")
    print("  • comparison_grid.jpg - Side-by-side comparison")
    print("  • filtered_photos/ - Your photo with all filters (if photo.jpg exists)")
    print("\nNext steps:")
    print("  1. Replace 'photo.jpg' with your own image")
    print("  2. Run: python demo.py")
    print("  3. Check the output images!")