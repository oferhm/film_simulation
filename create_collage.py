#!/usr/bin/env python3
"""
Standalone script to create a photo collage from filtered images.
This script can be run independently to create a collage from existing filtered photos.
"""

import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from film_filters.photo_collage import PhotoCollage


def main():
    """Main function to create the collage."""
    print("🖼️  Film Filter Photo Collage Creator")
    print("📁 Creating collage from photos in output directory...")
    print("=" * 60)
    
    try:
        # Create and run collage
        collage_creator = PhotoCollage()
        success = collage_creator.create_collage()
        
        if success:
            print("\n🎉 Collage created successfully!")
        else:
            print("\n❌ Failed to create collage!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()