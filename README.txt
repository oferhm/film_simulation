# Film Filters Project

A Python project for applying various film-style filters to photos and creating photo collages.

## Features

### Interactive GUI Application
- 🎨 Real-time filter preview and switching
- 📁 Easy photo loading with file browser
- 👀 Original/Filtered comparison view
- 🖼️ Large photo display with proper scaling
- 🌙 Professional dark theme interface
- ⚡ Multi-threaded filter processing (no GUI freezing)

### Film Filters
- Multiple vintage and analog film filters
- Kodak Portra, Ektar 100, ColorPlus 200 emulations
- Ilford HP5+ black and white film styles
- Vintage color grading effects
- Expired film aesthetic

### Photo Collage
- Automatically creates a 4x4 grid collage of all filtered photos
- Smart image resizing while maintaining aspect ratios
- Labeled photos with filter names
- Professional layout with spacing and background

## Installation

1. Activate your virtual environment:
   .\.venv\Scripts\Activate.ps1

2. Install dependencies:
   pip install -r requirements.txt

## Usage

### Method 1: Interactive GUI (Recommended)
Launch the interactive GUI application for real-time filter preview:

.\.venv\Scripts\python.exe film_filter_gui.py

OR use the launcher:

.\.venv\Scripts\python.exe launch_gui.py

GUI Features:
- 📁 Load photos with file browser
- 🖼️ Large photo display with zoom support  
- 🎨 One-click filter application with live preview
- �🎨 Original/Filtered comparison buttons
- 🎯 Real-time filter switching without saving
- 📱 Modern dark theme interface

### Method 2: Batch Processing + Auto Collage
Run all filters and automatically create a collage afterward:

.\.venv\Scripts\python.exe run_all_filters.py

This will:
1. Apply all available filters to your input photo
2. Save individual filtered photos to output/ directory  
3. Automatically create a photo collage with all results

run specific filter:
film_filters> .\.venv\Scripts\python.exe .\film_filters\filters\expired_film.py

### Method 3: Create Collage Only
If you already have filtered photos and just want to create a new collage:

.\.venv\Scripts\python.exe create_collage.py

## File Structure

- input/photo.jpg - Place your source photo here
- output/ - Contains all filtered photos and the final collage
- film_filters/filters/ - Individual filter scripts
- film_filters/photo_collage.py - Collage creation class
- config.py - Output file paths configuration

## Output

- Individual filtered photos saved as: output/[filter_name].jpg
- Final photo collage saved as: output/photo_collage.jpg

## Customization

You can modify the collage settings in film_filters/photo_collage.py:
- Grid size (default: 4x4)
- Cell dimensions (default: 400x300)
- Spacing between photos
- Background color
- Font styling for labels
