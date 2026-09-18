import cv2
import numpy as np
import torch
from PIL import Image

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


def unique_image_colors(image):
    rgb8 = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    unique_codes, inverse, counts = torch.unique(color_codes.flatten(), return_inverse=True, return_counts=True)
    unique_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=image.dtype) / 255.0
    unique_oklab = srgb_to_oklab(unique_rgb)
    return unique_codes, inverse, counts, unique_oklab


def map_unique_colors(image, inverse, unique_oklab, palette_oklab, palette_rgb):
    labels = nearest_palette_labels(unique_oklab, palette_oklab)
    reduced_rgb = palette_rgb[labels[inverse]].reshape(image.shape[0], image.shape[1], 3)
    if image.shape[-1] == 3:
        return reduced_rgb
    return torch.cat((reduced_rgb, image[..., 3:]), dim=-1)


def reduce_image_palette(image, color_count):
    unique_codes, inverse, counts, unique_oklab = unique_image_colors(image)
    if unique_codes.shape[0] > PALETTE_TRAINING_COLORS:
        positions = torch.linspace(0, inverse.shape[0] - 1, PALETTE_TRAINING_COLORS, device=image.device).round().long()
        training_indices, training_counts = torch.unique(inverse[positions], return_counts=True)
        training_oklab = unique_oklab[training_indices]
    else:
        training_oklab = unique_oklab
        training_counts = counts
    palette_oklab = perceptual_palette(training_oklab, training_counts, color_count)
    palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
    return map_unique_colors(image, inverse, unique_oklab, palette_oklab, palette_rgb)


def shared_perceptual_palette(images, color_count):
    colors, counts = sampled_image_colors(images)
    return perceptual_palette(colors, counts, color_count)


def palette_from_image(palette_image, additional_color, color_count, dtype, device):
    palette_pixels = palette_image[..., :3].to(device=device, dtype=dtype).reshape(-1, 3)
    rgb8 = (palette_pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes, counts = torch.unique(color_codes, return_counts=True)
    palette_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=dtype) / 255.0
    if unique_codes.shape[0] > color_count:
        training_rgb = palette_rgb
        training_counts = counts
        if unique_codes.shape[0] > PALETTE_TRAINING_COLORS:
            positions = torch.linspace(0, color_codes.shape[0] - 1, PALETTE_TRAINING_COLORS,
                                       dtype=torch.float64, device=device).round().long()
            training_codes, training_counts = torch.unique(color_codes[positions], return_counts=True)
            training_rgb = torch.stack((
                (training_codes >> 16) & 255,
                (training_codes >> 8) & 255,
                training_codes & 255,
            ), dim=1).to(dtype=dtype) / 255.0
        palette_oklab = perceptual_palette(srgb_to_oklab(training_rgb), training_counts, color_count)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0

    if additional_color is not None:
        palette_rgb = torch.cat((palette_rgb, additional_color.to(device=device, dtype=dtype).reshape(1, 3)))
    rgb8 = (palette_rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes = torch.unique(color_codes)
    palette_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=dtype) / 255.0
    return palette_rgb, srgb_to_oklab(palette_rgb)


def fixed_palette_from_image(palette_image, background_color, color_count, dtype, device, preserve_background):
    reference_rgb, reference_oklab = palette_from_image(palette_image, None, color_count, dtype, device)
    background = (background_color.to(device=device, dtype=dtype).clamp(0.0, 1.0) * 255.0).round() / 255.0 if preserve_background else None
    if background is None:
        return reference_rgb, reference_oklab, reference_oklab, None
    palette_rgb = torch.cat((reference_rgb, background.reshape(1, 3)))
    rgb8 = (palette_rgb.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes = torch.unique(color_codes)
    palette_rgb = torch.stack(((unique_codes >> 16) & 255, (unique_codes >> 8) & 255, unique_codes & 255), dim=1).to(dtype=dtype) / 255.0
    return palette_rgb, srgb_to_oklab(palette_rgb), reference_oklab, background


def sampled_image_colors(images, excluded_color=None):
    pixels = images[..., :3].reshape(-1, 3)
    sample_count = min(PALETTE_TRAINING_COLORS, pixels.shape[0])
    if sample_count < pixels.shape[0]:
        positions = torch.linspace(0, pixels.shape[0] - 1, sample_count, dtype=torch.float64, device=images.device).round().long()
        pixels = pixels[positions]
    rgb8 = (pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    if excluded_color is not None:
        excluded8 = (excluded_color.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
        included = torch.any(rgb8 != excluded8, dim=1)
        if torch.any(included):
            rgb8 = rgb8[included]
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes, counts = torch.unique(color_codes, return_counts=True)
    unique_rgb = torch.stack(((unique_codes >> 16) & 255, (unique_codes >> 8) & 255, unique_codes & 255), dim=1).to(dtype=images.dtype) / 255.0
    return srgb_to_oklab(unique_rgb), counts


def palette_alignment_error(colors, counts, palette, lightness_offset):
    total = torch.zeros((), dtype=colors.dtype, device=colors.device)
    weights = counts.to(dtype=colors.dtype)
    for start in range(0, colors.shape[0], PALETTE_ASSIGNMENT_CHUNK):
        chunk = colors[start:start + PALETTE_ASSIGNMENT_CHUNK]
        lightness = chunk[:, None, 0] + lightness_offset - palette[None, :, 0]
        chroma = chunk[:, None, 1:] - palette[None, :, 1:]
        distances = lightness.square() + chroma.square().sum(dim=-1)
        total += (distances.amin(dim=1) * weights[start:start + chunk.shape[0]]).sum()
    return total / weights.sum()


def estimate_palette_lightness_offset(images, palette, excluded_color=None):
    colors, counts = sampled_image_colors(images, excluded_color)
    offsets = torch.linspace(-0.2, 0.2, 17, dtype=colors.dtype, device=colors.device)
    errors = torch.stack([palette_alignment_error(colors, counts, palette, offset) for offset in offsets])
    best = offsets[errors.argmin()]
    offsets = torch.linspace(best - 0.025, best + 0.025, 9, dtype=colors.dtype, device=colors.device).clamp(-0.25, 0.25)
    errors = torch.stack([palette_alignment_error(colors, counts, palette, offset) for offset in offsets])
    return offsets[errors.argmin()]


def reduce_image_batch(images, color_count, shared_palette=False, fixed_palette=None, align_palette_lightness=False):
    if fixed_palette is None and not shared_palette:
        return torch.stack([reduce_image_palette(image, color_count) for image in images])

    if fixed_palette is None:
        palette_oklab = shared_perceptual_palette(images, color_count)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
    else:
        palette_rgb, palette_oklab = fixed_palette[:2]

    lightness_offset = None
    background = None
    if fixed_palette is not None and len(fixed_palette) == 4:
        reference_oklab = fixed_palette[2]
        background = fixed_palette[3]
        if align_palette_lightness:
            lightness_offset = estimate_palette_lightness_offset(images, reference_oklab, background)

    output = torch.empty_like(images)
    for index, image in enumerate(images):
        _, inverse, _, unique_oklab = unique_image_colors(image)
        matching_oklab = unique_oklab
        if lightness_offset is not None:
            matching_oklab = unique_oklab.clone()
            matching_oklab[:, 0] += lightness_offset
        mapped = map_unique_colors(image, inverse, matching_oklab, palette_oklab, palette_rgb)
        if background is not None:
            rgb8 = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
            background8 = (background.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
            mask = torch.all(rgb8 == background8, dim=-1, keepdim=True)
            mapped_rgb = torch.where(mask, background.reshape(1, 1, 3), mapped[..., :3])
            mapped = mapped_rgb if mapped.shape[-1] == 3 else torch.cat((mapped_rgb, mapped[..., 3:]), dim=-1)
        output[index] = mapped
    return output


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


def detect_axis_offsets(edge_energy, size, logical_size):
    candidate_count = max(1, (size + logical_size - 1) // logical_size)
    if logical_size == 1 or candidate_count == 1:
        return torch.zeros(edge_energy.shape[0], dtype=torch.long, device=edge_energy.device)

    period = max(1, round(size / logical_size))
    periodic_energy = torch.zeros_like(edge_energy)
    if period < edge_energy.shape[1]:
        paired_energy = (edge_energy[:, :-period] * edge_energy[:, period:]).sqrt()
        periodic_energy[:, :-period] += paired_energy
        periodic_energy[:, period:] += paired_energy
    else:
        periodic_energy = edge_energy

    boundaries = torch.arange(logical_size, device=edge_energy.device) * size // logical_size
    scores = []
    for offset in range(candidate_count):
        positions = (boundaries + offset) % size
        positions = positions[positions > 0]
        scores.append(periodic_energy[:, positions - 1].mean(dim=1))
    return torch.stack(scores, dim=1).argmax(dim=1)


def detect_pixel_grid_offsets(image, logical_width, logical_height):
    rgb = image[..., :3]
    horizontal_edges = (rgb[:, :, 1:] - rgb[:, :, :-1]).abs().mean(dim=(1, 3))
    vertical_edges = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(dim=(2, 3))
    column_offsets = detect_axis_offsets(horizontal_edges, image.shape[2], logical_width)
    row_offsets = detect_axis_offsets(vertical_edges, image.shape[1], logical_height)
    return column_offsets, row_offsets


def collapse_single_image_grid(image, logical_width, logical_height, scale_to_original, allow_uneven_grid,
                               column_offset, row_offset):
    height, width, channels = image.shape
    if column_offset or row_offset:
        image = torch.roll(image, shifts=(-row_offset, -column_offset), dims=(0, 1))
    if allow_uneven_grid and (height % logical_height != 0 or width % logical_width != 0):
        row_bounds = torch.arange(logical_height + 1, device=image.device) * height // logical_height
        column_bounds = torch.arange(logical_width + 1, device=image.device) * width // logical_width
        row_ids = torch.repeat_interleave(torch.arange(logical_height, device=image.device), row_bounds[1:] - row_bounds[:-1])
        column_ids = torch.repeat_interleave(torch.arange(logical_width, device=image.device), column_bounds[1:] - column_bounds[:-1])
        cell_count = logical_height * logical_width
        cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
        center_rows = row_bounds[:-1] + (row_bounds[1:] - row_bounds[:-1]) // 2
        center_columns = column_bounds[:-1] + (column_bounds[1:] - column_bounds[:-1]) // 2
        center_pixels = image[center_rows[:, None], center_columns[None, :], :].reshape(-1, channels)
        selected = select_modal_pixels(image.reshape(-1, channels), cell_ids, cell_count, center_pixels)
        logical = selected.reshape(logical_height, logical_width, channels)
        if scale_to_original:
            output = logical.index_select(0, row_ids).index_select(1, column_ids)
    else:
        block_height = height // logical_height
        block_width = width // logical_width
        block_pixels = block_height * block_width
        cells = image.reshape(logical_height, block_height, logical_width, block_width, channels)
        cells = cells.permute(0, 2, 1, 3, 4).reshape(-1, block_pixels, channels)
        cell_count = cells.shape[0]
        cell_ids = torch.arange(cell_count, device=image.device)[:, None].expand(cell_count, block_pixels).reshape(-1)
        center_index = (block_height // 2) * block_width + block_width // 2
        selected = select_modal_pixels(cells.reshape(-1, channels), cell_ids, cell_count, cells[:, center_index])
        logical = selected.reshape(logical_height, logical_width, channels)
        if scale_to_original:
            output = logical.repeat_interleave(block_height, dim=0).repeat_interleave(block_width, dim=1)

    if not scale_to_original:
        return logical
    if column_offset or row_offset:
        output = torch.roll(output, shifts=(row_offset, column_offset), dims=(0, 1))
    return output


def collapse_pixel_grid_mode(image, logical_width, logical_height, scale_to_original=True, allow_uneven_grid=False,
                             x_offset=0.0, y_offset=0.0, detected_offsets=None):
    batch_size, height, width, channels = image.shape
    if detected_offsets is None:
        column_offsets = [int(x_offset * width / logical_width + 0.5)] * batch_size
        row_offsets = [int(y_offset * height / logical_height + 0.5)] * batch_size
    else:
        column_offsets = detected_offsets[0].tolist()
        row_offsets = detected_offsets[1].tolist()

    output_height = height if scale_to_original else logical_height
    output_width = width if scale_to_original else logical_width
    output = torch.empty((batch_size, output_height, output_width, channels), dtype=image.dtype, device=image.device)
    for index, batch_image in enumerate(image):
        output[index] = collapse_single_image_grid(batch_image, logical_width, logical_height, scale_to_original,
                                                   allow_uneven_grid, column_offsets[index], row_offsets[index])
    return output


def frame_edge_map(frame, scale):
    rgb = (frame[..., :3].detach().to(device="cpu").clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    if frame.shape[-1] > 3:
        alpha = frame[..., 3].detach().to(device="cpu").numpy()
        rgb[alpha < 0.5] = 255
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    if rgb.shape[0] <= 4 or rgb.shape[1] <= 4:
        raise ValueError("MiniMax H3 pixel art autorefiner requires images larger than 4x4 pixels")
    grey = np.asarray(Image.fromarray(rgb[2:-2, 2:-2]).convert("L"))
    edges = cv2.Canny(grey, 50, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def cluster_mesh_lines(lines, threshold=4):
    lines = sorted(lines)
    clusters = [[lines[0]]]
    for position in lines[1:]:
        if position - clusters[-1][-1] <= threshold:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [int(np.median(cluster)) for cluster in clusters]


def detect_mesh_lines(edges):
    detected = cv2.HoughLinesP(
        edges, 1.0, np.deg2rad(1.0), 100, minLineLength=50, maxLineGap=10
    )
    height, width = edges.shape
    lines_x = [0, width - 1]
    lines_y = [0, height - 1]
    if detected is not None:
        for x1, y1, x2, y2 in detected.reshape(-1, 4):
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle > np.deg2rad(75):
                lines_x.append(round((x1 + x2) / 2))
            elif angle < np.deg2rad(15):
                lines_y.append(round((y1 + y2) / 2))
    return cluster_mesh_lines(lines_x), cluster_mesh_lines(lines_y)


def estimate_mesh_pixel_width(mesh):
    gaps = np.concatenate([np.diff(lines) for lines in mesh])
    low = np.percentile(gaps, 20)
    high = np.percentile(gaps, 80)
    middle = gaps[(gaps >= low) & (gaps <= high)]
    return max(1, int(np.round(np.median(middle if len(middle) else gaps))))


def filter_weak_mesh_lines(lines, edges, pixel_width):
    lines = list(lines)
    radius = max(1, round(pixel_width * 0.2))
    while len(lines) > 2:
        support = [np.count_nonzero(edges[:, max(0, position - radius):position + radius + 1]) for position in lines]
        remove = []
        for index in range(1, len(lines) - 1):
            close = min(lines[index] - lines[index - 1], lines[index + 1] - lines[index]) < pixel_width * 0.75
            surrounding_support = (support[index - 1] + support[index + 1]) / 2
            if close and support[index] < surrounding_support * 0.7:
                remove.append(index)
        if not remove:
            break
        lines = [position for index, position in enumerate(lines) if index not in remove]
    return lines


def homogenize_mesh_lines(lines, pixel_width):
    completed = []
    for start, end in zip(lines, lines[1:]):
        cell_count = max(1, int(np.round((end - start) / pixel_width)))
        cell_width = (end - start) / cell_count
        completed.extend(start + int(index * cell_width) for index in range(cell_count))
    completed.append(lines[-1])
    return completed


def mesh_from_edges(edges):
    initial = detect_mesh_lines(edges)
    if len(initial[0]) in (2, 3) and len(initial[1]) in (2, 3):
        return initial
    pixel_width = estimate_mesh_pixel_width(initial)
    lines_x = filter_weak_mesh_lines(initial[0], edges, pixel_width)
    lines_y = filter_weak_mesh_lines(initial[1], edges.T, pixel_width)
    return homogenize_mesh_lines(lines_x, pixel_width), homogenize_mesh_lines(lines_y, pixel_width)


def usable_mesh(mesh):
    lines_x, lines_y = mesh
    return len(lines_x) >= 2 and len(lines_y) >= 2 and not (len(lines_x) in (2, 3) and len(lines_y) in (2, 3))


def regular_mesh(image_width, image_height, logical_width, logical_height):
    lines_x = [round(index * image_width / logical_width) for index in range(logical_width + 1)]
    lines_y = [round(index * image_height / logical_height) for index in range(logical_height + 1)]
    return lines_x, lines_y


def detect_pixel_mesh(frame, fallback_width, fallback_height):
    mesh = mesh_from_edges(frame_edge_map(frame, 2))
    if usable_mesh(mesh):
        return mesh, 2
    mesh = mesh_from_edges(frame_edge_map(frame, 1))
    if usable_mesh(mesh):
        return mesh, 1
    return regular_mesh(frame.shape[1], frame.shape[0], fallback_width, fallback_height), 1


def select_modal_rgba(pixels, cell_ids, cell_count, center_pixels):
    rgba8 = (pixels[..., :4].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgba8[..., 0] << 24) | (rgba8[..., 1] << 16) | (rgba8[..., 2] << 8) | rgba8[..., 3]
    color_keys = (cell_ids << 32) | color_codes
    unique_keys, key_inverse, counts = torch.unique(color_keys, return_inverse=True, return_counts=True)
    unique_cells = unique_keys >> 32
    unique_colors = unique_keys & 0xFFFFFFFF

    max_counts = torch.zeros(cell_count, dtype=counts.dtype, device=pixels.device)
    max_counts.scatter_reduce_(0, unique_cells, counts, reduce="amax")
    modal = counts == max_counts[unique_cells]

    candidate_rgba = torch.stack((
        (unique_colors >> 24) & 255,
        (unique_colors >> 16) & 255,
        (unique_colors >> 8) & 255,
        unique_colors & 255,
    ), dim=1).to(dtype=pixels.dtype) / 255.0
    center_oklab = srgb_to_oklab(center_pixels[..., :3].clamp(0.0, 1.0))
    candidate_oklab = srgb_to_oklab(candidate_rgba[:, :3])
    distances = (candidate_oklab - center_oklab[unique_cells]).square().sum(dim=1)
    distances += (candidate_rgba[:, 3] - center_pixels[unique_cells, 3]).square()
    modal_distances = torch.where(modal, distances, torch.inf)
    min_distances = torch.full((cell_count,), torch.inf, dtype=pixels.dtype, device=pixels.device)
    min_distances.scatter_reduce_(0, unique_cells, modal_distances, reduce="amin")
    closest = modal & torch.isclose(distances, min_distances[unique_cells], rtol=1e-5, atol=1e-8)

    eligible = closest[key_inverse]
    positions = torch.arange(pixels.shape[0], device=pixels.device)
    first_positions = torch.full((cell_count,), pixels.shape[0], dtype=torch.long, device=pixels.device)
    first_positions.scatter_reduce_(0, cell_ids, torch.where(eligible, positions, pixels.shape[0]), reduce="amin")
    return pixels[first_positions]


def collapse_frame_mesh(frame, mesh, scale):
    rgb8 = (frame[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    background = rgb8[0, 0]
    visible = torch.any(rgb8 != background, dim=-1)
    if frame.shape[-1] > 3:
        visible &= frame[..., 3] > 0
    rgba = torch.zeros((*frame.shape[:2], 4), dtype=frame.dtype, device=frame.device)
    rgba[..., :3] = frame[..., :3]
    rgba[..., 3] = visible.to(dtype=frame.dtype)
    if scale > 1:
        rgba = rgba.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)

    lines_x, lines_y = mesh
    row_lengths = torch.tensor(np.diff(lines_y), device=frame.device)
    column_lengths = torch.tensor(np.diff(lines_x), device=frame.device)
    logical_height = len(lines_y) - 1
    logical_width = len(lines_x) - 1
    row_ids = torch.repeat_interleave(torch.arange(logical_height, device=frame.device), row_lengths)
    column_ids = torch.repeat_interleave(torch.arange(logical_width, device=frame.device), column_lengths)
    pixels = rgba[:lines_y[-1], :lines_x[-1]]
    cell_ids = (row_ids[:, None] * logical_width + column_ids[None, :]).reshape(-1)
    center_rows = torch.tensor(lines_y[:-1], device=frame.device) + row_lengths // 2
    center_columns = torch.tensor(lines_x[:-1], device=frame.device) + column_lengths // 2
    center_pixels = rgba[center_rows[:, None], center_columns[None, :]].reshape(-1, 4)
    selected = select_modal_rgba(pixels.reshape(-1, 4), cell_ids, logical_height * logical_width, center_pixels)
    return selected.reshape(logical_height, logical_width, 4)


def fit_frame_to_size(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    scaled_width = max(1, min(width, round(frame.shape[1] * scale)))
    scaled_height = max(1, min(height, round(frame.shape[0] * scale)))
    scaled = torch.nn.functional.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0), size=(scaled_height, scaled_width), mode="nearest-exact"
    )[0].permute(1, 2, 0)
    output = torch.zeros((height, width, 4), dtype=frame.dtype, device=frame.device)
    top = (height - scaled_height) // 2
    left = (width - scaled_width) // 2
    output[top:top + scaled_height, left:left + scaled_width] = scaled
    return output


def composite_white(image):
    alpha = image[..., 3:4].clamp(0.0, 1.0)
    return image[..., :3] * alpha + (1.0 - alpha)


def split_animation_holds(images, transition_threshold):
    rgb8 = (images[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int16)
    differences = (rgb8[1:] - rgb8[:-1]).abs().to(torch.float32).mean(dim=(1, 2, 3))
    boundaries = [index + 1 for index, difference in enumerate(differences.tolist()) if difference >= transition_threshold]
    starts = [0] + boundaries
    ends = boundaries + [images.shape[0]]
    return [images[start:end] for start, end in zip(starts, ends)]


def animation_hold_representative(hold):
    frames = (hold[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.float32).reshape(hold.shape[0], -1)
    scores = torch.zeros(hold.shape[0], device=hold.device)
    for frame in frames:
        scores += (frames - frame).abs().mean(dim=1)
    return frames[scores.argmin()]


def detect_animation_pose_count(holds):
    if len(holds) == 1:
        return 1
    representatives = torch.stack([animation_hold_representative(hold) for hold in holds])
    distances = torch.empty((len(holds), len(holds)), device=representatives.device)
    for index, representative in enumerate(representatives):
        distances[index] = (representatives - representative).abs().mean(dim=1)
    distances = distances.tolist()

    for pose_count in range(2, len(holds) // 2 + 1):
        if len(holds) % pose_count:
            continue
        matches = True
        for index in range(pose_count, len(holds)):
            phase = index % pose_count
            wrong_distance = min(distances[index][candidate] for candidate in range(pose_count) if candidate != phase)
            if distances[index][phase] * 2 >= wrong_distance:
                matches = False
                break
        if matches:
            return pose_count
    raise ValueError(
        f"Could not find a repeated cycle among {len(holds)} held poses. "
        "Set pose_count explicitly, adjust transition_threshold, or provide at least two complete cycles."
    )


def modal_rgb_composite(frames):
    frame_count, height, width = frames.shape[:3]
    rgb8 = (frames[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    pixel_count = height * width
    pixel_ids = torch.arange(pixel_count, device=frames.device).repeat(frame_count)
    color_keys = (pixel_ids << 24) | color_codes.reshape(-1)
    unique_keys, counts = torch.unique(color_keys, return_counts=True)
    unique_pixels = unique_keys >> 24
    unique_colors = unique_keys & 0xFFFFFF

    max_counts = torch.zeros(pixel_count, dtype=counts.dtype, device=frames.device)
    max_counts.scatter_reduce_(0, unique_pixels, counts, reduce="amax")
    modal = counts == max_counts[unique_pixels]

    red = (unique_colors >> 16) & 255
    green = (unique_colors >> 8) & 255
    blue = unique_colors & 255
    luma = 2126 * red + 7152 * green + 722 * blue
    rank = luma * 0x1000000 + unique_colors
    sentinel = torch.iinfo(torch.int64).max
    selected = torch.full((pixel_count,), sentinel, dtype=torch.int64, device=frames.device)
    selected.scatter_reduce_(0, unique_pixels, torch.where(modal, rank, sentinel), reduce="amin")
    selected &= 0xFFFFFF
    rgb = torch.stack(((selected >> 16) & 255, (selected >> 8) & 255, selected & 255), dim=1)
    return rgb.to(dtype=frames.dtype).reshape(height, width, 3) / 255.0


class PixelArtAnimationPoseCompositor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PixelArtAnimationPoseCompositor",
            display_name="Pixel Art Animation Pose Compositor",
            description="Separates held animation poses by frame difference and combines repeated cycles with an exact RGB mode.",
            category="image/animation",
            inputs=[
                io.Image.Input("images"),
                io.Int.Input("pose_count", default=0, min=0, max=256, step=1,
                             tooltip="Number of distinct poses in one animation cycle. Set to 0 to autodetect from repeated cycles."),
                io.Float.Input("transition_threshold", default=7.0, min=0.0, max=255.0, step=0.1,
                               tooltip="Minimum mean absolute 8-bit RGB difference between consecutive frames that starts a new held pose."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, images, pose_count, transition_threshold):
        if images.shape[-1] < 3:
            raise ValueError(f"Animation pose compositing requires at least 3 image channels, got {images.shape[-1]}")
        holds = split_animation_holds(images, transition_threshold)
        if pose_count == 0:
            pose_count = detect_animation_pose_count(holds)
        if len(holds) < pose_count or len(holds) % pose_count:
            raise ValueError(
                f"Detected {len(holds)} held poses; expected a positive multiple of pose_count={pose_count}. "
                "Adjust transition_threshold or trim the input batch."
            )
        composites = [modal_rgb_composite(torch.cat(holds[pose::pose_count])) for pose in range(pose_count)]
        return io.NodeOutput(torch.stack(composites))


class MiniMaxH3PixelArtRefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PixelArtRefiner",
            display_name="MiniMax H3 Pixel Art Refiner",
            description="Combines perceptual palette reduction with a modal pixel-grid collapse in a selectable order.",
            category="image/minimax",
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
                io.Boolean.Input("shared_palette", default=False,
                                 tooltip="Generate one palette from the entire image batch instead of a separate palette for each image."),
                io.Float.Input("x_offset", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Shift the grid right by this fraction of one logical pixel."),
                io.Float.Input("y_offset", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Shift the grid down by this fraction of one logical pixel."),
                io.Boolean.Input("auto_offset", default=False,
                                 tooltip="Detect the X and Y grid phase separately for each image. Overrides x_offset and y_offset."),
                io.Boolean.Input("align_palette_lightness", default=True,
                                 tooltip="Compensate for a shared OKLab lightness shift before matching a supplied palette."),
                io.Boolean.Input("preserve_generated_background", default=True,
                                 tooltip="Keep the first input image's upper-left color instead of forcing it into the supplied palette."),
                io.Image.Input("palette_image", optional=True,
                               tooltip="Use this image's colors as the palette for the entire batch. Palettes above colors are reduced first. Overrides shared_palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, color_reduction_first, scale_to_original, allow_uneven_grid=False,
                shared_palette=False, palette_image=None, x_offset=0.0, y_offset=0.0, auto_offset=False,
                align_palette_lightness=True, preserve_generated_background=True):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"MiniMax H3 pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if palette_image is not None and palette_image.shape[-1] < 3:
            raise ValueError(f"Palette image requires at least 3 image channels, got {palette_image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")
        if not allow_uneven_grid and (image_width % width != 0 or image_height % height != 0):
            raise ValueError(f"Image size {image_width}x{image_height} must be divisible by logical grid {width}x{height}")

        fixed_palette = None
        if palette_image is not None:
            fixed_palette = fixed_palette_from_image(palette_image, image[0, 0, 0, :3], colors, image.dtype, image.device,
                                                     preserve_generated_background)
        detected_offsets = detect_pixel_grid_offsets(image, width, height) if auto_offset else None

        if color_reduction_first:
            reduced = reduce_image_batch(image, colors, shared_palette, fixed_palette, align_palette_lightness)
            return io.NodeOutput(collapse_pixel_grid_mode(reduced, width, height, scale_to_original, allow_uneven_grid,
                                                          x_offset, y_offset, detected_offsets))

        collapsed = collapse_pixel_grid_mode(image, width, height, scale_to_original, allow_uneven_grid, x_offset, y_offset,
                                             detected_offsets)
        return io.NodeOutput(reduce_image_batch(collapsed, colors, shared_palette, fixed_palette, align_palette_lightness))


class MiniMaxH3PixelArtAutorefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PixelArtAutorefiner",
            display_name="MiniMax H3 Pixel Art Autorefiner",
            description="Detects or applies one pixel mesh across the full frame without splitting sprites.",
            category="image/minimax",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Full-frame logical width for manual mode or automatic detection fallback."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Full-frame logical height for manual mode or automatic detection fallback."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum generated or supplied palette size."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Scale and pad the collapsed frame to the input dimensions."),
                io.Boolean.Input("shared_palette", default=True,
                                 tooltip="Generate one palette from the entire image batch instead of a separate palette for each frame."),
                io.Boolean.Input("manual_resolution", default=False,
                                 tooltip="Divide the full frame evenly into width × height cells instead of detecting a mesh."),
                io.Boolean.Input("align_palette_lightness", default=True,
                                 tooltip="Compensate for a shared OKLab lightness shift before matching a supplied palette."),
                io.Boolean.Input("preserve_generated_background", default=True,
                                 tooltip="Keep the first input image's upper-left color instead of forcing it into the supplied palette."),
                io.Image.Input("palette_image", optional=True,
                               tooltip="Use this image's colors as the palette for the entire batch. Overrides shared_palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, scale_to_original, shared_palette=False, palette_image=None,
                manual_resolution=False, align_palette_lightness=True, preserve_generated_background=True):
        if image.shape[-1] < 3:
            raise ValueError(f"MiniMax H3 pixel art autorefiner requires at least 3 image channels, got {image.shape[-1]}")
        if palette_image is not None and palette_image.shape[-1] < 3:
            raise ValueError(f"Palette image requires at least 3 image channels, got {palette_image.shape[-1]}")
        if width > image.shape[2] or height > image.shape[1]:
            raise ValueError(f"Fallback grid {width}x{height} cannot exceed image size {image.shape[2]}x{image.shape[1]}")

        fixed_palette = None
        if palette_image is not None:
            fixed_palette = fixed_palette_from_image(palette_image, image[0, 0, 0, :3], colors, image.dtype, image.device,
                                                     preserve_generated_background)
        if manual_resolution:
            mesh = regular_mesh(image.shape[2], image.shape[1], width, height)
            mesh_scale = 1
        else:
            mesh, mesh_scale = detect_pixel_mesh(image[0], width, height)
        reduced = reduce_image_batch(image, colors, shared_palette, fixed_palette, align_palette_lightness)
        collapsed = [collapse_frame_mesh(frame, mesh, mesh_scale) for frame in reduced]

        if scale_to_original:
            output = [fit_frame_to_size(frame, image.shape[2], image.shape[1]) for frame in collapsed]
            return io.NodeOutput(composite_white(torch.stack(output)))
        return io.NodeOutput(composite_white(torch.stack(collapsed)))


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
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, color_reduction_first, scale_to_original):
        image_height, image_width = image.shape[1:3]
        if image.shape[-1] < 3:
            raise ValueError(f"Krea 2 pixel art refinement requires at least 3 image channels, got {image.shape[-1]}")
        if width > image_width or height > image_height:
            raise ValueError(f"Logical grid {width}x{height} cannot exceed image size {image_width}x{image_height}")
        if image_width % width != 0 or image_height % height != 0:
            raise ValueError(f"Image size {image_width}x{image_height} must be divisible by logical grid {width}x{height}")

        if color_reduction_first:
            reduced = torch.stack([reduce_image_palette(batch_image, colors) for batch_image in image])
            return io.NodeOutput(collapse_pixel_grid_mode(reduced, width, height, scale_to_original))

        collapsed = collapse_pixel_grid_mode(image, width, height, scale_to_original)
        return io.NodeOutput(torch.stack([reduce_image_palette(batch_image, colors) for batch_image in collapsed]))
