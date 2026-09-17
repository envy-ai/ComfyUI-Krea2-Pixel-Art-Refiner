import torch

from comfy_api.latest import io


PALETTE_TRAINING_COLORS = 65536
PALETTE_ASSIGNMENT_CHUNK = 16384


def srgb_to_oklab(rgb):
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    red, green, blue = linear.unbind(-1)
    l = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l, m, s = l.pow(1.0 / 3.0), m.pow(1.0 / 3.0), s.pow(1.0 / 3.0)
    return torch.stack((
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ), dim=-1)


def oklab_to_srgb(lab):
    lightness, a, b = lab.unbind(-1)
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b).pow(3)
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b).pow(3)
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b).pow(3)
    red = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    green = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    blue = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    linear = torch.stack((red, green, blue), dim=-1)
    return torch.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)


def nearest_palette_labels(colors, palette):
    labels = torch.empty(colors.shape[0], dtype=torch.long, device=colors.device)
    for start in range(0, colors.shape[0], PALETTE_ASSIGNMENT_CHUNK):
        chunk = colors[start:start + PALETTE_ASSIGNMENT_CHUNK]
        distances = (chunk[:, None, :] - palette[None, :, :]).square().sum(dim=-1)
        labels[start:start + chunk.shape[0]] = distances.argmin(dim=1)
    return labels


def perceptual_palette(colors, counts, color_count):
    color_count = min(color_count, colors.shape[0])
    weights = counts.to(dtype=colors.dtype)
    first = weights.argmax()
    palette = colors[first].unsqueeze(0)
    min_distances = (colors - palette[0]).square().sum(dim=1)
    for _ in range(1, color_count):
        next_color = (min_distances * weights.sqrt()).argmax()
        new_center = colors[next_color]
        palette = torch.cat((palette, new_center.unsqueeze(0)))
        distance = (colors - new_center).square().sum(dim=1)
        min_distances = torch.minimum(min_distances, distance)

    for _ in range(12):
        labels = nearest_palette_labels(colors, palette)
        weighted_colors = colors * weights[:, None]
        sums = torch.zeros_like(palette)
        totals = torch.zeros(color_count, dtype=weights.dtype, device=weights.device)
        sums.scatter_add_(0, labels[:, None].expand(-1, 3), weighted_colors)
        totals.scatter_add_(0, labels, weights)
        palette = torch.where(totals[:, None] > 0, sums / totals.clamp_min(1.0)[:, None], palette)
    return palette


def reduce_image_palette(image, color_count):
    rgb8 = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    unique_codes, inverse, counts = torch.unique(color_codes.flatten(), return_inverse=True, return_counts=True)
    unique_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=image.dtype) / 255.0
    unique_oklab = srgb_to_oklab(unique_rgb)
    if unique_codes.shape[0] > PALETTE_TRAINING_COLORS:
        positions = torch.linspace(0, inverse.shape[0] - 1, PALETTE_TRAINING_COLORS, device=image.device).round().long()
        training_indices, training_counts = torch.unique(inverse[positions], return_counts=True)
        training_oklab = unique_oklab[training_indices]
    else:
        training_oklab = unique_oklab
        training_counts = counts
    palette_oklab = perceptual_palette(training_oklab, training_counts, color_count)
    labels = nearest_palette_labels(unique_oklab, palette_oklab)
    palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
    reduced_rgb = palette_rgb[labels[inverse]].reshape(image.shape[0], image.shape[1], 3)
    if image.shape[-1] == 3:
        return reduced_rgb
    return torch.cat((reduced_rgb, image[..., 3:]), dim=-1)


def select_modal_pixels(pixels, cell_ids, cell_count, center_pixels):
    rgb8 = (pixels[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    color_keys = (cell_ids << 24) | color_codes
    unique_keys, key_inverse, counts = torch.unique(color_keys, return_inverse=True, return_counts=True)
    unique_cells = unique_keys >> 24
    unique_colors = unique_keys & 0xFFFFFF

    max_counts = torch.zeros(cell_count, dtype=counts.dtype, device=pixels.device)
    max_counts.scatter_reduce_(0, unique_cells, counts, reduce="amax")
    modal = counts == max_counts[unique_cells]

    candidate_rgb = torch.stack((
        (unique_colors >> 16) & 255,
        (unique_colors >> 8) & 255,
        unique_colors & 255,
    ), dim=1).to(dtype=pixels.dtype) / 255.0
    center_oklab = srgb_to_oklab(center_pixels[..., :3].clamp(0.0, 1.0))
    candidate_oklab = srgb_to_oklab(candidate_rgb)
    distances = (candidate_oklab - center_oklab[unique_cells]).square().sum(dim=1)
    modal_distances = torch.where(modal, distances, torch.inf)
    min_distances = torch.full((cell_count,), torch.inf, dtype=pixels.dtype, device=pixels.device)
    min_distances.scatter_reduce_(0, unique_cells, modal_distances, reduce="amin")
    closest = modal & torch.isclose(distances, min_distances[unique_cells], rtol=1e-5, atol=1e-8)

    eligible = closest[key_inverse]
    positions = torch.arange(pixels.shape[0], device=pixels.device)
    first_positions = torch.full((cell_count,), pixels.shape[0], dtype=torch.long, device=pixels.device)
    first_positions.scatter_reduce_(0, cell_ids, torch.where(eligible, positions, pixels.shape[0]), reduce="amin")
    return pixels[first_positions]


def collapse_pixel_grid_mode(image, logical_width, logical_height, scale_to_original=True, allow_uneven_grid=False):
    batch_size, height, width, channels = image.shape
    if allow_uneven_grid and (height % logical_height != 0 or width % logical_width != 0):
        row_bounds = torch.arange(logical_height + 1, device=image.device) * height // logical_height
        column_bounds = torch.arange(logical_width + 1, device=image.device) * width // logical_width
        row_ids = torch.repeat_interleave(torch.arange(logical_height, device=image.device), row_bounds[1:] - row_bounds[:-1])
        column_ids = torch.repeat_interleave(torch.arange(logical_width, device=image.device), column_bounds[1:] - column_bounds[:-1])
        cells_per_image = logical_height * logical_width
        image_cells = row_ids[:, None] * logical_width + column_ids[None, :]
        batch_cells = torch.arange(batch_size, device=image.device)[:, None, None] * cells_per_image
        cell_ids = (batch_cells + image_cells).reshape(-1)
        center_rows = row_bounds[:-1] + (row_bounds[1:] - row_bounds[:-1]) // 2
        center_columns = column_bounds[:-1] + (column_bounds[1:] - column_bounds[:-1]) // 2
        center_pixels = image[:, center_rows[:, None], center_columns[None, :], :].reshape(-1, channels)
        selected = select_modal_pixels(image.reshape(-1, channels), cell_ids, batch_size * cells_per_image, center_pixels)
        logical = selected.reshape(batch_size, logical_height, logical_width, channels)
        if not scale_to_original:
            return logical
        return logical.index_select(1, row_ids).index_select(2, column_ids)

    block_height = height // logical_height
    block_width = width // logical_width
    block_pixels = block_height * block_width
    cells = image.reshape(batch_size, logical_height, block_height, logical_width, block_width, channels)
    cells = cells.permute(0, 1, 3, 2, 4, 5).reshape(-1, block_pixels, channels)
    cell_count = cells.shape[0]
    cell_ids = torch.arange(cell_count, device=image.device)[:, None].expand(cell_count, block_pixels).reshape(-1)
    center_index = (block_height // 2) * block_width + block_width // 2
    selected = select_modal_pixels(cells.reshape(-1, channels), cell_ids, cell_count, cells[:, center_index])
    logical = selected.reshape(batch_size, logical_height, logical_width, channels)
    if not scale_to_original:
        return logical
    return logical.repeat_interleave(block_height, dim=1).repeat_interleave(block_width, dim=2)


class Krea2PixelArtRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2PixelArtRefiner",
            display_name="Krea 2 Pixel Art Refiner",
            description="Combines perceptual palette reduction with a modal pixel-grid collapse in a selectable order.",
            category="image/krea2",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels across the refined image."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Number of logical pixels down the refined image."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum number of colors in each image's palette."),
                io.Boolean.Input("color_reduction_first", default=True,
                                 tooltip="Reduce the palette before grid collapse. Disable for collapse-then-reduce behavior."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Expand the logical pixel grid back to the input dimensions."),
                io.Boolean.Input("allow_uneven_grid", default=False,
                                 tooltip="Allow input dimensions that are not evenly divisible by the logical grid."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, color_reduction_first, scale_to_original, allow_uneven_grid=False):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"Krea 2 pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")
        if not allow_uneven_grid and (image_width % width != 0 or image_height % height != 0):
            raise ValueError(f"Image size {image_width}x{image_height} must be divisible by logical grid {width}x{height}")

        if color_reduction_first:
            reduced = torch.stack([reduce_image_palette(batch_image, colors) for batch_image in image])
            return io.NodeOutput(collapse_pixel_grid_mode(reduced, width, height, scale_to_original, allow_uneven_grid))

        collapsed = collapse_pixel_grid_mode(image, width, height, scale_to_original, allow_uneven_grid)
        return io.NodeOutput(torch.stack([reduce_image_palette(batch_image, colors) for batch_image in collapsed]))
