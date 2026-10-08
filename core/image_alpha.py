"""Flatten transparent images to RGB without losing their shape."""
from PIL import Image

LIGHT_BG = (255, 255, 255)
DARK_BG = (20, 20, 20)


def flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB; transparent pixels go onto a background that contrasts with the content.

    `convert("RGB")` keeps the colour stored under transparent pixels, which for many
    logo PNGs is the logo's own colour: the logo turns into a solid rectangle. A dark
    logo gets a light background and a light logo a dark one. Opaque images are a plain
    conversion.
    """
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    if not has_alpha:
        return img if img.mode == "RGB" else img.convert("RGB")

    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] == 255:
        return rgba.convert("RGB")

    # Mean luminance of the visible pixels decides the background.
    visible = rgba.convert("L").histogram(mask=alpha.point(lambda a: 255 if a > 127 else 0))
    n = sum(visible)
    mean = sum(i * c for i, c in enumerate(visible)) / n if n else 255
    bg = LIGHT_BG if mean < 128 else DARK_BG
    out = Image.new("RGB", rgba.size, bg)
    out.paste(rgba, mask=alpha)
    return out
