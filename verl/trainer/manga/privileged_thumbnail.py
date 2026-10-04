"""Shared training and audit rendering: position hints, no image replacement."""
from functools import lru_cache
from PIL import ImageDraw, ImageFont
from . import scene_graph as sg


@lru_cache(maxsize=1)
def _label_font():
    # Pillow searches system font directories on both Linux and Windows.
    for name in ('DejaVuSans-Bold.ttf', 'arialbd.ttf', 'LiberationSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, 18)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=18)
    except TypeError:  # Older Pillow fallback; supported training installs use TrueType.
        return ImageFont.load_default()


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def render_thumbnail(original, boxes, kind):
    result = original.convert('RGB').copy()
    result.thumbnail((448, 448))
    draw = ImageDraw.Draw(result)
    rectangles = [sg._pixel_box(box, *result.size) for box in boxes]
    for rectangle in rectangles:
        draw.rectangle(rectangle, outline='red', width=2)
    font = _label_font()
    placed = []
    for i, rectangle in enumerate(rectangles):
        label = chr(65+i) if kind == 'link' else str(i+1)
        bounds = draw.textbbox((0, 0), label, font=font)
        width = min(result.width, bounds[2]-bounds[0]+10)
        height = min(result.height, bounds[3]-bounds[1]+8)
        x0, y0, x1, y1 = rectangle
        # Prefer outside the upper-left corner. Clamp every alternative to the
        # image, then avoid previously placed labels and covering target regions.
        origins = [(x0, y0-height-2), (x1-width, y0-height-2),
                   (x0, y1+2), (x1+2, y0), (x0-width-2, y0),
                   (x0+2, y0+2), (x1-width-2, y1-height-2)]
        candidates = []
        for x, y in origins:
            x = max(0, min(result.width-width, x))
            y = max(0, min(result.height-height, y))
            candidates.append((x, y, x+width, y+height))
        badge = min(candidates, key=lambda r: (
            sum(_overlap(r, old) for old in placed),
            sum(_overlap(r, box) for box in rectangles)))
        placed.append(badge)
        draw.rounded_rectangle(badge, radius=3, fill='#a00020', outline='white', width=1)
        draw.text((badge[0]+5-bounds[0], badge[1]+4-bounds[1]), label,
                  font=font, fill='white')
    return result
