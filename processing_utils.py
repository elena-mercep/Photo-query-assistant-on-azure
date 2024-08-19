from PIL import Image


def resize_image(image_path, output_path, resize_factor):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            new_size = (int(width * resize_factor), int(height * resize_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(output_path)
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")

