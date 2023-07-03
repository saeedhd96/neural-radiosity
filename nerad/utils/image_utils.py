from pathlib import Path

import mitsuba as mi
from torch import Tensor


def save_image(root: str, name: str, formats: list[str], image):
    """Save image to multiple formats, each in a sub folder"""
    assert all(fmt in {"png", "exr"} for fmt in formats)

    root: Path = Path(root)
    for fmt in formats:
        folder = root / fmt
        folder.mkdir(parents=True, exist_ok=True)
        write_bitmap(str(folder / f"{name}.{fmt}"), image)


def convert_to_bitmap(data, gamma_correction, uint8_srgb=True):
    """
    Convert the RGB image in `data` to a `Bitmap`. `uint8_srgb` defines whether
    the resulting bitmap should be translated to a uint8 sRGB bitmap.
    """

    if isinstance(data, mi.Bitmap):
        bitmap = data
    else:
        if isinstance(data, Tensor):
            data = data.detach().cpu().numpy()
        bitmap = mi.Bitmap(data)

    if uint8_srgb:
        bitmap = bitmap.convert(
            mi.Bitmap.PixelFormat.RGBA,
            mi.Struct.Type.UInt8,
            gamma_correction,
        )

    return bitmap


def write_bitmap(filename, data, tonemap=True,  write_async=True, quality=-1):
    """
    Write the RGB image in `data` to a PNG/EXR/.. file.
    """
    uint8_srgb = Path(filename).suffix in {".png", ".jpg", ".jpeg", ".webp"}

    bitmap = convert_to_bitmap(data, tonemap, uint8_srgb)

    if write_async:
        bitmap.write_async(filename, quality=quality)
    else:
        bitmap.write(filename, quality=quality)
