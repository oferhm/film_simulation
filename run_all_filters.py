import os
import subprocess
import sys
import time
from pathlib import Path
from film_filters.photo_collage import PhotoCollage

# Get the correct Python executable (from virtual environment)
python_exe = sys.executable
print(f"Using Python: {python_exe}")
print("=" * 60)

# Specify filters directory
folder_path = './film_filters/filters'

# Check if directory exists
if not os.path.exists(folder_path):
    print(f"❌ Directory not found: {folder_path}")
    exit(1)

# Get all Python files
python_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]

if not python_files:
    print(f"❌ No Python files found in {folder_path}")
    exit(1)

print(f"Found {len(python_files)} Python files to run:")
for i, filename in enumerate(python_files, 1):
    print(f"  {i}. {filename}")

print("\n" + "=" * 60)

# Run each file
success_count = 0
failed_files = []

for i, filename in enumerate(python_files, 1):
    file_path = os.path.join(folder_path, filename)
    print(f"\n[{i}/{len(python_files)}] Running {filename}...")
    
    start_time = time.time()
    
    try:
        # Run the file with proper Python executable
        result = subprocess.run([python_exe, file_path], 
                              capture_output=True, 
                              text=True, 
                              timeout=60)  # 60 second timeout
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"  ✅ Success ({duration:.2f}s)")
            success_count += 1
            
            # Show any output
            if result.stdout.strip():
                print(f"     Output: {result.stdout.strip()}")
                
        else:
            print(f"  ❌ Failed ({duration:.2f}s)")
            failed_files.append(filename)
            
            if result.stderr.strip():
                print(f"     Error: {result.stderr.strip()}")
                
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Timeout (>60s)")
        failed_files.append(filename)
        
    except Exception as e:
        print(f"  💥 Exception: {str(e)}")
        failed_files.append(filename)

# Summary
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"  ✅ Successful: {success_count}/{len(python_files)}")
print(f"  ❌ Failed: {len(failed_files)}/{len(python_files)}")

if failed_files:
    print(f"\nFailed files:")
    for filename in failed_files:
        print(f"  • {filename}")
else:
    print("\n🎉 All filters ran successfully!")

print("=" * 60)

# Create photo collage if all filters succeeded
if success_count > 0:
    print("\n🖼️  Creating photo collage...")
    try:
        collage_creator = PhotoCollage()
        collage_success = collage_creator.create_collage()
        
        if collage_success:
            print("✨ Photo collage created successfully!")
        else:
            print("⚠️  Collage creation failed, but filters completed successfully.")
            
    except Exception as e:
        print(f"❌ Error creating collage: {e}")
        print("⚠️  Filters completed successfully, but collage creation failed.")

print("\n" + "=" * 60)