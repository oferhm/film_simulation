"""
Film Simulation Filters
Implements 5 popular film looks: Portra, Fuji, Disposable, Cinematic, Vintage
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FilmSimulator:
    """Base class for film simulation filters"""
    
    def __init__(self, image_path: str = None, image: np.ndarray = None):
        """
        Initialize with either image path or numpy array
        
        Args:
            image_path: Path to image file
            image: NumPy array of image (BGR format)
        """
        if image_path:
            self.original = cv2.imread(image_path)
            if self.original is None:
                raise ValueError(f"Could not load image from {image_path}")
        elif image is not None:
            self.original = image.copy()
        else:
            raise ValueError("Must provide either image_path or image")
        
        self.image = self.original.copy()
        self.height, self.width = self.image.shape[:2]
    
    def reset(self):
        """Reset to original image"""
        self.image = self.original.copy()
        return self
    
    def adjust_temperature(self, kelvin: int) -> 'FilmSimulator':
        """
        Adjust color temperature
        
        Args:
            kelvin: Temperature shift (-500 to +500)
        """
        # Convert to float
        img = self.image.astype(np.float32)
        
        # Positive = warmer (more red/yellow), Negative = cooler (more blue)
        if kelvin > 0:
            # Warm: increase red, decrease blue
            factor = kelvin / 500.0
            img[:, :, 2] = np.clip(img[:, :, 2] * (1 + factor * 0.3), 0, 255)  # Red
            img[:, :, 0] = np.clip(img[:, :, 0] * (1 - factor * 0.2), 0, 255)  # Blue
        else:
            # Cool: increase blue, decrease red
            factor = abs(kelvin) / 500.0
            img[:, :, 0] = np.clip(img[:, :, 0] * (1 + factor * 0.3), 0, 255)  # Blue
            img[:, :, 2] = np.clip(img[:, :, 2] * (1 - factor * 0.2), 0, 255)  # Red
        
        self.image = img.astype(np.uint8)
        return self
    
    def adjust_tint(self, value: int) -> 'FilmSimulator':
        """
        Adjust green/magenta tint
        
        Args:
            value: Tint adjustment (-20 to +20)
        """
        img = self.image.astype(np.float32)
        
        if value > 0:
            # More green
            factor = value / 20.0
            img[:, :, 1] = np.clip(img[:, :, 1] * (1 + factor * 0.2), 0, 255)
        else:
            # More magenta
            factor = abs(value) / 20.0
            img[:, :, 1] = np.clip(img[:, :, 1] * (1 - factor * 0.2), 0, 255)
        
        self.image = img.astype(np.uint8)
        return self
    
    def adjust_contrast(self, value: int) -> 'FilmSimulator':
        """
        Adjust contrast
        
        Args:
            value: Contrast adjustment (-50 to +50)
        """
        factor = (259 * (value + 255)) / (255 * (259 - value))
        self.image = cv2.convertScaleAbs(self.image, alpha=factor, beta=128*(1-factor))
        return self
    
    def adjust_highlights(self, value: int) -> 'FilmSimulator':
        """
        Adjust highlights
        
        Args:
            value: Highlight adjustment (-100 to +100)
        """
        # Create mask for highlights (bright areas)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 180).astype(np.float32)
        
        # Apply adjustment to highlights
        factor = 1 + (value / 100.0)
        img = self.image.astype(np.float32)
        
        for i in range(3):
            img[:, :, i] = img[:, :, i] * (1 - mask) + img[:, :, i] * factor * mask
        
        self.image = np.clip(img, 0, 255).astype(np.uint8)
        return self
    
    def adjust_shadows(self, value: int) -> 'FilmSimulator':
        """
        Adjust shadows
        
        Args:
            value: Shadow adjustment (-100 to +100)
        """
        # Create mask for shadows (dark areas)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        mask = (gray < 75).astype(np.float32)
        
        # Apply adjustment to shadows
        factor = 1 + (value / 100.0)
        img = self.image.astype(np.float32)
        
        for i in range(3):
            img[:, :, i] = img[:, :, i] * (1 - mask) + img[:, :, i] * factor * mask
        
        self.image = np.clip(img, 0, 255).astype(np.uint8)
        return self
    
    def adjust_blacks(self, value: int) -> 'FilmSimulator':
        """
        Adjust black point (fade)
        
        Args:
            value: Black adjustment (0 to +50)
        """
        # Lift black point
        img = self.image.astype(np.float32)
        img = img + value
        self.image = np.clip(img, 0, 255).astype(np.uint8)
        return self
    
    def adjust_saturation(self, value: int) -> 'FilmSimulator':
        """
        Adjust saturation
        
        Args:
            value: Saturation adjustment (-100 to +100)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation
        factor = 1 + (value / 100.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self
    
    def adjust_hsl_color(self, color_mask: str, hue: int = 0, sat: int = 0, lum: int = 0) -> 'FilmSimulator':
        """
        Adjust specific color range in HSL
        
        Args:
            color_mask: Color to adjust ('red', 'orange', 'yellow', 'green', 'blue')
            hue: Hue shift (-30 to +30)
            sat: Saturation adjustment (-50 to +50)
            lum: Luminance adjustment (-50 to +50)
        """
        # Define color ranges in HSV
        color_ranges = {
            'red': ((0, 10), (170, 180)),    # Red wraps around
            'orange': (10, 25),
            'yellow': (25, 35),
            'green': (35, 85),
            'blue': (85, 130),
        }
        
        if color_mask not in color_ranges:
            return self
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Create mask for the color
        if color_mask == 'red':
            # Red is special (wraps around hue wheel)
            mask1 = cv2.inRange(hsv[:, :, 0], color_ranges['red'][0][0], color_ranges['red'][0][1])
            mask2 = cv2.inRange(hsv[:, :, 0], color_ranges['red'][1][0], color_ranges['red'][1][1])
            mask = cv2.bitwise_or(mask1, mask2).astype(np.float32) / 255.0
        else:
            lower, upper = color_ranges[color_mask]
            mask = cv2.inRange(hsv[:, :, 0], lower, upper).astype(np.float32) / 255.0
        
        # Apply adjustments to masked area
        if hue != 0:
            hsv[:, :, 0] = hsv[:, :, 0] * (1 - mask) + (hsv[:, :, 0] + hue) * mask
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        
        if sat != 0:
            factor = 1 + (sat / 100.0)
            hsv[:, :, 1] = hsv[:, :, 1] * (1 - mask) + hsv[:, :, 1] * factor * mask
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        if lum != 0:
            factor = 1 + (lum / 100.0)
            hsv[:, :, 2] = hsv[:, :, 2] * (1 - mask) + hsv[:, :, 2] * factor * mask
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self
    
    def add_grain(self, amount: int = 25, size: int = 20) -> 'FilmSimulator':
        """
        Add film grain
        
        Args:
            amount: Grain intensity (0-100)
            size: Grain size (10-30)
        """
        # Create grain
        grain = np.random.normal(0, amount, (self.height, self.width)).astype(np.float32)
        
        # Blur grain to control size
        kernel_size = max(1, size // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        grain = cv2.GaussianBlur(grain, (kernel_size, kernel_size), 0)
        
        # Add grain to image
        img = self.image.astype(np.float32)
        for i in range(3):
            img[:, :, i] = img[:, :, i] + grain
        
        self.image = np.clip(img, 0, 255).astype(np.uint8)
        return self
    
    def add_vignette(self, strength: int = 30) -> 'FilmSimulator':
        """
        Add vignette effect
        
        Args:
            strength: Vignette strength (0-100)
        """
        # Create radial gradient
        rows, cols = self.height, self.width
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Center point
        center_x, center_y = cols / 2, rows / 2
        
        # Calculate distance from center
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Create vignette mask (darker at edges)
        vignette = 1 - (distance / max_distance) * (strength / 100.0)
        vignette = np.clip(vignette, 0, 1)
        
        # Apply vignette
        img = self.image.astype(np.float32)
        for i in range(3):
            img[:, :, i] = img[:, :, i] * vignette
        
        self.image = np.clip(img, 0, 255).astype(np.uint8)
        return self
    
    def adjust_clarity(self, value: int) -> 'FilmSimulator':
        """
        Adjust clarity (micro-contrast)
        
        Args:
            value: Clarity adjustment (-50 to +50)
        """
        # Convert to LAB
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Apply unsharp mask to L channel
        l_channel = lab[:, :, 0]
        blurred = cv2.GaussianBlur(l_channel, (0, 0), 10)
        
        if value > 0:
            # Increase clarity (sharpen)
            lab[:, :, 0] = l_channel + (l_channel - blurred) * (value / 50.0)
        else:
            # Decrease clarity (soften)
            lab[:, :, 0] = l_channel - (l_channel - blurred) * (abs(value) / 50.0)
        
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        self.image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return self
    
    def split_tone(self, shadow_hue: int, highlight_hue: int, balance: float = 0.5) -> 'FilmSimulator':
        """
        Apply split toning (different colors to shadows and highlights)
        
        Args:
            shadow_hue: Hue for shadows (0-360)
            highlight_hue: Hue for highlights (0-360)
            balance: Balance between shadows and highlights (0-1)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Create luminance mask
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Shadows mask (dark areas)
        shadow_mask = 1 - gray
        shadow_mask = np.power(shadow_mask, 2)  # Make it more selective
        
        # Highlights mask (bright areas)
        highlight_mask = gray
        highlight_mask = np.power(highlight_mask, 2)
        
        # Apply split tone
        # Shadows
        shadow_influence = shadow_mask * (1 - balance) * 0.3
        hsv[:, :, 0] = hsv[:, :, 0] * (1 - shadow_influence) + (shadow_hue / 2) * shadow_influence
        
        # Highlights
        highlight_influence = highlight_mask * balance * 0.3
        hsv[:, :, 0] = hsv[:, :, 0] * (1 - highlight_influence) + (highlight_hue / 2) * highlight_influence
        
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
        
        self.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return self
    
    def get_result(self) -> np.ndarray:
        """Return the processed image"""
        return self.image


# ==================== FILM PRESETS ====================

def kodak_portra(image_path: str = None, image: np.ndarray = None) -> np.ndarray:
    """
    Kodak Portra 400 Look
    - Warm skin tones
    - Soft contrast
    - Pastel colors
    - Lifted shadows
    """
    sim = FilmSimulator(image_path=image_path, image=image)
    
    # Tone adjustments
    sim.adjust_highlights(-30)
    sim.adjust_shadows(20)
    sim.adjust_blacks(10)
    
    # Color adjustments
    sim.adjust_temperature(200)
    sim.adjust_saturation(-10)
    
    # HSL adjustments
    sim.adjust_hsl_color('red', hue=5, sat=-5, lum=5)
    sim.adjust_hsl_color('orange', hue=5, sat=-5, lum=10)
    sim.adjust_hsl_color('green', hue=20, sat=-20, lum=10)
    sim.adjust_hsl_color('blue', hue=-10, sat=-10, lum=5)
    
    # Add grain
    sim.add_grain(amount=25, size=20)
    
    return sim.get_result()


def fuji_film(image_path: str = None, image: np.ndarray = None) -> np.ndarray:
    """
    Fujifilm Pro 400H Look
    - Cooler tones
    - Strong greens
    - Deep blues
    - Slightly higher contrast
    """
    sim = FilmSimulator(image_path=image_path, image=image)
    
    # Tone adjustments
    sim.adjust_contrast(10)
    sim.adjust_highlights(-20)
    
    # Color adjustments
    sim.adjust_temperature(-150)
    sim.adjust_tint(-5)
    
    # HSL adjustments
    sim.adjust_hsl_color('green', hue=-15, sat=10)
    sim.adjust_hsl_color('blue', hue=-10, sat=15)
    sim.adjust_hsl_color('yellow', hue=-5, sat=10)
    
    # Add grain
    sim.add_grain(amount=20, size=15)
    
    return sim.get_result()


def disposable_camera(image_path: str = None, image: np.ndarray = None) -> np.ndarray:
    """
    Disposable Camera Look (Kodak ColorPlus 200)
    - Strong flash look
    - High grain
    - Saturated colors
    - Strong vignette
    """
    sim = FilmSimulator(image_path=image_path, image=image)
    
    # Tone adjustments
    sim.adjust_contrast(25)
    sim.adjust_highlights(-40)
    
    # Color adjustments
    sim.adjust_saturation(15)
    
    # Effects
    sim.add_grain(amount=40, size=25)
    sim.add_vignette(strength=30)
    sim.adjust_clarity(-10)
    
    return sim.get_result()


def cinematic_film(image_path: str = None, image: np.ndarray = None) -> np.ndarray:
    """
    Cinematic Film Look (Kodak Vision3 500T)
    - Teal shadows
    - Warm highlights
    - Soft contrast
    - Halation glow
    """
    sim = FilmSimulator(image_path=image_path, image=image)
    
    # Split toning: teal shadows (200°), orange highlights (35°)
    sim.split_tone(shadow_hue=200, highlight_hue=35, balance=0.5)
    
    # Soft contrast
    sim.adjust_contrast(-5)
    sim.adjust_highlights(-15)
    sim.adjust_shadows(10)
    
    # Slight glow effect (reduce clarity)
    sim.adjust_clarity(-15)
    
    # Add grain
    sim.add_grain(amount=20, size=18)
    
    return sim.get_result()


def vintage_film(image_path: str = None, image: np.ndarray = None) -> np.ndarray:
    """
    Vintage Film Look (Kodak Gold 200)
    - Warm yellow tones
    - Faded blacks
    - Soft contrast
    - Nostalgic feel
    """
    sim = FilmSimulator(image_path=image_path, image=image)
    
    # Fade effect
    sim.adjust_blacks(15)
    sim.adjust_shadows(15)
    sim.adjust_highlights(-20)
    
    # Warm yellow tones
    sim.adjust_temperature(300)
    sim.adjust_tint(5)
    
    # Desaturate
    sim.adjust_saturation(-15)
    
    # Reduce contrast
    sim.adjust_contrast(-10)
    
    # Add grain
    sim.add_grain(amount=30, size=22)
    
    return sim.get_result()


# ==================== DEMO / USAGE ====================

def apply_all_filters(image_path: str, output_dir: str = "output/"):
    """
    Apply all 5 film filters to an image and save results
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save original
    cv2.imwrite(f"{output_dir}{base_name}_original.jpg", original)
    
    # Apply filters
    filters = {
        'portra': kodak_portra,
        'fuji': fuji_film,
        'disposable': disposable_camera,
        'cinematic': cinematic_film,
        'vintage': vintage_film
    }
    
    for name, filter_func in filters.items():
        print(f"Applying {name} filter...")
        result = filter_func(image=original)
        output_path = f"{output_dir}{base_name}_{name}.jpg"
        cv2.imwrite(output_path, result)
        print(f"  Saved: {output_path}")
    
    print(f"\n✅ All filters applied! Check {output_dir}")


# Example usage
if __name__ == "__main__":
    # Example 1: Apply single filter
    # result = kodak_portra(image_path="photo.jpg")
    # cv2.imwrite("output_portra.jpg", result)
    
    # Example 2: Apply all filters
    # apply_all_filters("photo.jpg", output_dir="filtered_photos/")
    
    # Example 3: Custom adjustments
    # sim = FilmSimulator(image_path="photo.jpg")
    # sim.adjust_temperature(200).adjust_saturation(-10).add_grain(25)
    # cv2.imwrite("custom_filter.jpg", sim.get_result())
    
    print("Film Simulation Filters Ready!")
    print("\nAvailable filters:")
    print("  1. kodak_portra() - Warm, soft, pastel (most popular)")
    print("  2. fuji_film() - Cool, strong greens and blues")
    print("  3. disposable_camera() - Nostalgic 90s look")
    print("  4. cinematic_film() - Teal & orange movie look")
    print("  5. vintage_film() - Warm, faded, retro")
    print("\nUsage:")
    print("  result = kodak_portra(image_path='photo.jpg')")
    print("  cv2.imwrite('output.jpg', result)")