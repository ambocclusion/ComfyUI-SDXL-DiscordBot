from PIL import Image


def crop(image: Image, x, y, width, height):
    """
    Crop the given image to the specified rectangle.

    Parameters:
    image (PIL.Image): The image to be cropped.
    x (int): The x-coordinate of the top-left corner of the crop rectangle.
    y (int): The y-coordinate of the top-left corner of the crop rectangle.
    width (int): The width of the crop rectangle.
    height (int): The height of the crop rectangle.

    Returns:
    PIL.Image: The cropped image.
    """
    return image.crop((x, y, x + width, y + height))


def smart_crop(image: Image):
    """
    Crop an image to either 16:9, 9:16, or 1:1 based on its dimensions.
    
    Parameters:
    image (PIL.Image): The image to be cropped.    
    
    Returns:
    PIL.Image: The cropped image.
    """
    width, height = image.size
    aspect_ratio = width / height

    if aspect_ratio > 1.77:  # Wider than 16:9
        new_width = int(height * 16 / 9)
        x_offset = (width - new_width) // 2
        return crop(image, x_offset, 0, new_width, height)
    elif aspect_ratio < 0.56:  # Taller than 9:16
        new_height = int(width * 16 / 9)
        y_offset = (height - new_height) // 2
        return crop(image, 0, y_offset, width, new_height)
    else:  # Between 9:16 and 16:9, crop to 1:1
        side_length = min(width, height)
        x_offset = (width - side_length) // 2
        y_offset = (height - side_length) // 2
        return crop(image, x_offset, y_offset, side_length, side_length)


def resize_image(image: Image, target_width, target_height):
    """
    Resize an image to the specified dimensions while maintaining aspect ratio.

    Parameters:
    image (PIL.Image): The image to be resized.
    target_width (int): The target width.
    target_height (int): The target height.

    Returns:
    PIL.Image: The resized image.
    """
    return image.resize((target_width, target_height))

def smart_resize(image: Image, target_max):
    """
    Resize an image to fit within a square of target_max x target_max while maintaining aspect ratio.

    Parameters:
    image (PIL.Image): The image to be resized.
    target_max (int): The maximum dimension (width or height).

    Returns:
    PIL.Image: The resized image.
    """
    width, height = image.size
    if width > height:
        new_width = target_max
        new_height = int((target_max / width) * height)
    else:
        new_height = target_max
        new_width = int((target_max / height) * width)
    
    return resize_image(image, new_width, new_height)
