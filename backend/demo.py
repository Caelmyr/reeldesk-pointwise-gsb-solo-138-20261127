"""Deterministic synthetic film and scanner controls for the teaching roll."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from .imaging import probe_original
from .values import Box, image_bytes, rgb_image


WIDTH = 1120
HEIGHT = 408
BASE = np.array([.91, .68, .43], dtype=np.float32)
DPI = 121.7
DEMO_CODE = "RD-041"
DEMO_MARKER = "synthetic-roll-2026-1"


@dataclass(frozen=True)
class DemoCapture:
    name: str
    role: str
    medium: str
    data: bytes
    exposure: str
    notes: str


def _smooth_noise(shape, block, random):
    height, width = shape
    coarse = random.normal(0, 1, size=(max(2, (height + block - 1) // block), max(2, (width + block - 1) // block)))
    yy, xx = np.meshgrid(np.linspace(0, coarse.shape[0] - 1, height),
                         np.linspace(0, coarse.shape[1] - 1, width), indexing="ij")
    expanded = ndimage.map_coordinates(coarse, [yy, xx], order=3, mode="reflect")
    return expanded


def _landscape(width, height, seed, variant):
    random = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    u = xx / (width - 1)
    v = yy / (height - 1)
    haze = _smooth_noise((height, width), 70, random)
    stones = _smooth_noise((height, width), 6, random)
    grass = _smooth_noise((height, width), 18, random)
    grain = random.normal(0, .016, size=(height, width))

    light_x = [.31, .75, .62][variant]
    light_y = [.38, .24, .27][variant]
    glow = np.exp(-((u - light_x) ** 2 * 4 + (v - light_y) ** 2 * 9))
    sky = .47 + glow * .29 - v * .12 + haze * .042
    clouds = (
        np.exp(-((v - .25 - .019 * haze) ** 2) * 510)
        + .6 * np.exp(-((v - .43 + .023 * haze) ** 2) * 340)
    ) * (.045 + .025 * grass)
    sky = sky + clouds

    height_line = (
        [.58, .51, .54][variant]
        - .104 * np.sin(2.2 * np.pi * u + variant * .7) ** 2
        - .082 * np.sin(4.4 * np.pi * u + variant * 1.1) ** 2
        + .018 * grass
    )
    mountain = .37 + glow * .04 + .011 * stones + .04 * haze
    shade_axis = np.sin(11 * np.pi * u + v * 12) + 0.24 * np.sin(23 * np.pi * u - v * 9)
    mountain -= np.maximum(shade_axis, 0) * np.maximum(0, v - height_line) * .33
    second_line = [.59, .53, .58][variant] - .095 * np.exp(-((u - [.87, .2, .8][variant]) ** 2) * 28) + .016 * grass
    terrace = .25 + .053 * haze - .025 * stones + 0.024 * np.sin(u * 39 + v * 23)
    image = np.where(v > height_line, mountain, sky)
    image = np.where(v > second_line, terrace, image)

    if variant == 0:
        shore = .62 + .14 * (u - .48) ** 2 - u * .045 + .014 * haze
        ripple = (.014 * np.sin(213 * v + 3 * u + haze) + .018 * np.sin(783 * v - u * 2.7)) * (.4 + u * .6)
        water = .34 + .19 * (1 - v) + ripple + glow * .14 + .025 * grass
        image = np.where(v > shore, water, image)
        for index in range(7):
            blade_x = .22 + (index - 3) * .028
            slope = (.1 + index * .011) * (v - .54)
            shape = (v > .66 - .006 * index) & (v < .91 + .006 * grass) & (np.abs(u - blade_x - slope) < .005 + .003 * v)
            image = np.where(shape, .12 + .05 * grass, image)
        wall = (v > .804 + .045 * np.sin(u * 9) + .026 * u) & (u < .48)
        image = np.where(wall, .14 + .065 * np.abs(stones) + .041 * grass, image)
    elif variant == 1:
        ground = .59 + .027 * np.sin(u * 11.9) + .014 * stones
        texture = .21 + .075 * stones + .032 * grass + .041 * haze
        image = np.where(v > ground, texture, image)
        left = .48
        right = .76
        roof = .48 + np.abs(u - .62) * .52
        shed = (u > left) & (u < right) & (v > roof) & (v < .743)
        roof_mask = shed & (v < roof + .041)
        body = .315 + .027 * (np.sin(u * 564) > .68) + .01 * stones
        image = np.where(shed, body, image)
        image = np.where(roof_mask, .15 + .02 * haze, image)
        pier = (u > .60) & (u < .66) & (v > .596) & (v < .744)
        image = np.where(pier, .095 + .03 * grass, image)
        pole = (np.abs(u - .22 - (v - .6) * .021) < .008) & (v > .32) & (v < .691)
        cable = (np.abs(v - (.37 + .06 * np.sin(3.8 * u))) < .0055) & (u < .28)
        image = np.where(pole | cable, .135 + .02 * haze, image)
    else:
        ground = .59 + .04 * np.sin(u * 8.6 + .5) + .014 * grass
        snow = .49 + .026 * grass + .045 * stones + glow * .13
        tracks = np.abs(u - .64 + (v - .56) * .97)
        trail = (tracks < .026 + v * .037) & (v > .56)
        trail_inside = (np.abs(tracks - (.018 + v * .022)) < .005 + .008 * v) & (v > .56)
        snow = np.where(trail, .29 + .092 * stones, snow)
        snow = np.where(trail_inside, .15 + .055 * stones, snow)
        image = np.where(v > ground, snow, image)
        for center, high in ((.22, .28), (.34, .47), (.853, .35), (.92, .399)):
            trunk = np.abs(u - center - (v - high) * .045) < .006
            tree = (v > high) & (v < .67) & (np.abs(u - center) < (v - high) * .226)
            branches = tree & (np.sin(179 * v + stones * 3.7) > -.46)
            image = np.where((trunk & (v > high) & (v < .70)) | branches, .133 + .035 * stones, image)
    image = np.clip(image + grain, .045, .92)
    tints = ([1.035, .98, .88], [1.01, .95, .83], [.92, 1, 1.07])[variant]
    rgb = np.clip(image[..., None] * np.asarray(tints, np.float32), .035, .94)
    return rgb.astype(np.float32)


FRAME_REGIONS = (
    (Box(63, 64, 310, 280), "01 · Breakwater"),
    (Box(405, 64, 310, 280), "02 · Relay shed"),
    (Box(748, 64, 310, 280), "03 · Winter road"),
)


def _negative_scene():
    random = np.random.default_rng(417)
    room = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    room[:] = BASE * .45
    film_border = Box(22, 24, WIDTH - 44, HEIGHT - 48)
    room[film_border.slices] = BASE
    for index, (region, _) in enumerate(FRAME_REGIONS):
        positive = _landscape(region.w, region.h, 155 + index * 151, index)
        density = .11 + 1.65 * positive ** 1.17
        room[region.slices] = BASE * 10 ** -density
    for x in range(36, WIDTH - 33, 23):
        y_positions = [38, HEIGHT - 50]
        for y in y_positions:
            room[y:y + 12, x:x + 12] = [.92, .94, .93]
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    grain = random.normal(0, .018, size=(HEIGHT, WIDTH, 1))
    stripes = (.0047 * np.sin(xx / 4.8) + .002 * np.sin(yy / 3.6))[..., None]
    return np.clip(room * (1 + grain) + stripes, 0, 1)


def _light_field(width, height, variant=0):
    yy, xx = np.mgrid[:height, :width]
    u = xx / (width - 1)
    v = yy / (height - 1)
    skew_x = .036 if variant else 0
    edge = ((u - .5 + skew_x) * 1.38) ** 2 + ((v - .5) * .88) ** 2
    band = .032 * np.sin(4.2 * np.pi * u + variant * .17) + .017 * np.cos(2.3 * np.pi * v)
    spot = .052 * np.exp(-((u - .28) ** 2 * 41 + (v - .32) ** 2 * 86))
    shade = .88 - .23 * edge + band - spot
    if variant:
        shade += .022 * np.sin(17 * v - 4 * u) - .018
    color = np.asarray([.99, 1.02, .963], np.float32)
    return (shade[..., None] * color).astype(np.float32)


def _dark_field(variant=0):
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    level = .0135 + .0055 * np.sin(xx / 271.0) + .0027 * np.cos(yy / 34)
    if variant:
        level += .0031
    return (level[..., None] * np.array([1.22, .93, 1.07])).astype(np.float32)


def _spot(mask, x, y, rx, ry, tilt):
    ys, xs = np.mgrid[:mask.shape[0], :mask.shape[1]]
    distance_x = xs - x - (ys - y) * tilt
    distance_y = ys - y
    return (distance_x / rx) ** 2 + (distance_y / ry) ** 2 < 1


def _sensor(scene, variant):
    dark = _dark_field(variant)
    flat = _light_field(WIDTH, HEIGHT, variant)
    strength = 1.04 if variant else .99
    result = dark + (flat - dark) * scene * strength
    random = np.random.default_rng(651 + variant)
    result += random.normal(0, .0016, size=result.shape)
    dust = [(163, 162, 3, 4, .2), (564, 143, 3.4, 3.7, .4), (581, 155, 3, 2, -.2),
            (930, 275, 2, 5, .25), (901, 284, 2.5, 2.7, .3)]
    if variant:
        dust = [(170, 230, 4, 3, .3), (510, 187, 3.3, 2, -.3), (983, 101, 2, 3.5, .25)]
    for x, y, rx, ry, tilt in dust:
        mask = _spot(result, x, y, rx, ry, tilt)
        result[mask] *= [ .44, .55, .61 ]
    clear = _spot(result, 678 if variant else 695, 177, 2.5, 2.9, -.2)
    result[clear] += .055
    return np.clip(result, 0, 1)


def _shift_scene(scene, dx, dy):
    return ndimage.shift(scene, (dy, dx, 0), order=1, mode="constant", cval=0)


def _scan_bytes(values):
    return image_bytes(rgb_image(values), dpi=(DPI, DPI), compress_level=6)


def _alpha_scan_bytes(values, variant):
    alpha = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    if variant:
        alpha[:, :13] = 0
        alpha[327:, 929:1002] = 0
        alpha[188:226, 582:635] = 0
    else:
        alpha[:, -10:] = 0
        alpha[64:105, 478:528] = 0
    image = rgb_image(values)
    image.putalpha(Image.fromarray(alpha))
    return image_bytes(image, dpi=(DPI, DPI), compress_level=6)


def captures():
    negative = _negative_scene()
    first = _sensor(negative, 0)
    second = _sensor(_shift_scene(negative, 9, -6), 1)
    return [
        DemoCapture("Coast survey · pass 01", "scan", "negative", _alpha_scan_bytes(first, 0), "0.9× integration",
                    "Synthetic orange-mask negative. Includes a few dust grains and scanner coverage gaps."),
        DemoCapture("Coast survey · pass 02", "scan", "negative", _alpha_scan_bytes(second, 1), "1.1× integration",
                    "Synthetic repeat pass shifted right 9 pixels and up 6 pixels, with changed dust and gaps."),
        DemoCapture("Backlight flat · 01", "flat", "control", _scan_bytes(_light_field(WIDTH, HEIGHT)), "1.0× clear field",
                    "Synthetic matching flat for pass 01."),
        DemoCapture("Shutter dark · 01", "dark", "control", _scan_bytes(_dark_field()), "Shuttered",
                    "Synthetic matching offset for pass 01."),
        DemoCapture("Backlight flat · 02", "flat", "control", _scan_bytes(_light_field(WIDTH, HEIGHT, 1)), "1.0× clear field",
                    "Synthetic matching flat for pass 02."),
        DemoCapture("Shutter dark · 02", "dark", "control", _scan_bytes(_dark_field(1)), "Shuttered",
                    "Synthetic matching offset for pass 02."),
    ]


def install_demo(catalog):
    with catalog.connection() as connection:
        previous = catalog.setting_in_transaction(connection, DEMO_MARKER)
    if previous:
        return previous
    images = captures()
    with catalog.transaction() as connection:
        previous = catalog.setting_in_transaction(connection, DEMO_MARKER)
        if previous:
            return previous
        roll = catalog.create_roll(
            {
                "title": "Coast survey",
                "stock": "35 mm color negative",
                "code": DEMO_CODE,
                "location": "Study archive · Drawer C",
                "notes": "A synthetic three-frame practice strip. It includes two scanner passes and matched controls.",
                "tags": ["demonstration", "color-negative"],
            },
            connection=connection,
        )
        images_out = []
        for index, capture in enumerate(images, 1):
            properties = {
                "name": capture.name,
                "original_name": f"coast-study-{index:02d}.png",
                "role": capture.role,
                "medium": capture.medium,
                "dpi": DPI,
                "scanner": "Simulated 135 film carrier",
                "exposure": capture.exposure,
                "notes": capture.notes,
            }
            image = catalog.add_scan(
                roll["id"], properties, capture.data, probe_original(capture.data), sample=True, connection=connection
            )
            images_out.append(image["id"])

        for index, scan_id in enumerate(images_out[:2]):
            dx, dy = (0, 0) if index == 0 else (9, -6)
            for region, label in FRAME_REGIONS:
                catalog.add_mark(
                    scan_id,
                    {"kind": "frame", "label": label, "geometry": region.translate(dx, dy).as_dict(), "notes": ""},
                    connection=connection,
                )
            sample_area = Box(378 + dx, 105 + dy, 20, 88)
            catalog.add_mark(
                scan_id,
                {"kind": "reference", "label": "Clear base", "geometry": sample_area.as_dict(), "notes": ""},
                connection=connection,
            )
            protection = Box(514 + dx, 181 + dy, 94, 64)
            catalog.add_mark(
                scan_id,
                {"kind": "keep", "label": "Shed edge detail", "geometry": protection.as_dict(), "notes": ""},
                connection=connection,
            )
        state = {"roll_id": roll["id"], "scans": images_out[:2], "controls": images_out[2:]}
        catalog.remember_setting(connection, DEMO_MARKER, state)
        return state
