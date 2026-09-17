import numpy as np
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
    pixels = images[..., :3].reshape(-1, 3)
    sample_count = min(PALETTE_TRAINING_COLORS, pixels.shape[0])
    if sample_count < pixels.shape[0]:
        positions = torch.linspace(0, pixels.shape[0] - 1, sample_count, dtype=torch.float64, device=images.device).round().long()
        pixels = pixels[positions]
    rgb8 = (pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[:, 0] << 16) | (rgb8[:, 1] << 8) | rgb8[:, 2]
    unique_codes, counts = torch.unique(color_codes, return_counts=True)
    unique_rgb = torch.stack((
        (unique_codes >> 16) & 255,
        (unique_codes >> 8) & 255,
        unique_codes & 255,
    ), dim=1).to(dtype=images.dtype) / 255.0
    return perceptual_palette(srgb_to_oklab(unique_rgb), counts, color_count)


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


def reduce_image_batch(images, color_count, shared_palette=False, fixed_palette=None):
    if fixed_palette is None and not shared_palette:
        return torch.stack([reduce_image_palette(image, color_count) for image in images])

    if fixed_palette is None:
        palette_oklab = shared_perceptual_palette(images, color_count)
        palette_rgb = (oklab_to_srgb(palette_oklab).clamp(0.0, 1.0) * 255.0).round() / 255.0
    else:
        palette_rgb, palette_oklab = fixed_palette

    output = torch.empty_like(images)
    for index, image in enumerate(images):
        _, inverse, _, unique_oklab = unique_image_colors(image)
        output[index] = map_unique_colors(image, inverse, unique_oklab, palette_oklab, palette_rgb)
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


def visible_components(mask, minimum_area=25):
    mask = mask.detach().to(device="cpu").numpy()
    parents = []
    runs = []
    previous = []

    def find(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    def union(first, second):
        first = find(first)
        second = find(second)
        if first != second:
            parents[max(first, second)] = min(first, second)

    for row_index, row in enumerate(mask):
        padded = np.pad(row, (1, 1))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        current = []
        previous_index = 0
        for start, end in zip(starts.tolist(), ends.tolist()):
            label = len(parents)
            parents.append(label)
            while previous_index < len(previous) and previous[previous_index][1] < start:
                previous_index += 1
            overlap_index = previous_index
            while overlap_index < len(previous) and previous[overlap_index][0] <= end:
                union(label, previous[overlap_index][2])
                overlap_index += 1
            current.append((start, end, label))
            runs.append((row_index, start, end, label))
        previous = current

    components = {}
    for row, start, end, label in runs:
        root = find(label)
        component = components.setdefault(root, {
            "area": 0,
            "top": row,
            "bottom": row + 1,
            "left": start,
            "right": end,
            "runs": [],
        })
        component["area"] += end - start
        component["top"] = min(component["top"], row)
        component["bottom"] = max(component["bottom"], row + 1)
        component["left"] = min(component["left"], start)
        component["right"] = max(component["right"], end)
        component["runs"].append((row, start, end))

    return sorted(
        (component for component in components.values() if component["area"] > minimum_area),
        key=lambda component: (component["top"], component["left"]),
    )


def component_mask(component, device):
    height = component["bottom"] - component["top"]
    width = component["right"] - component["left"]
    mask = np.zeros((height, width), dtype=np.bool_)
    for row, start, end in component["runs"]:
        mask[row - component["top"], start - component["left"]:end - component["left"]] = True
    return torch.from_numpy(mask).to(device=device)


def estimate_pixel_period(color_codes, maximum_logical_size):
    color_codes = color_codes.detach().to(device="cpu").numpy()
    size = color_codes.shape[1]
    minimum_period = max(1, (size + maximum_logical_size - 1) // maximum_logical_size)
    if size < 4:
        return minimum_period, 0

    run_lengths = []
    for row in color_codes:
        starts = np.concatenate(([0], np.flatnonzero(row[1:] != row[:-1]) + 1))
        ends = np.concatenate((starts[1:], [size]))
        run_lengths.extend((ends - starts)[row[starts] >= 0].tolist())
    if not run_lengths:
        return 1, 0

    counts = np.bincount(run_lengths, minlength=size + 1)
    weighted_counts = counts * np.arange(counts.shape[0])
    maximum_period = max(minimum_period, min(size // 2, counts.shape[0] - 1))
    period = 1
    for candidate in range(2, maximum_period + 1):
        previous = weighted_counts[candidate - 1]
        following = weighted_counts[candidate + 1] if candidate < maximum_period else 0
        if counts[candidate] > 1 and weighted_counts[candidate] > previous and weighted_counts[candidate] >= following:
            period = candidate
            break
    if period == 1 and minimum_period == 1:
        return 1, 0
    period = max(period, minimum_period)

    edges = (color_codes[:, 1:] != color_codes[:, :-1]).sum(axis=0)
    best_period = period
    best_correlation = -1.0
    for candidate in range(max(2, minimum_period, period - 1), min(maximum_period, period + 1) + 1):
        first = edges[:-candidate]
        second = edges[candidate:]
        denominator = np.sqrt(np.dot(first, first) * np.dot(second, second))
        correlation = np.dot(first, second) / denominator if denominator > 0 else 0.0
        if correlation > best_correlation:
            best_period = candidate
            best_correlation = correlation
    period = best_period
    positions = np.flatnonzero(edges > 0) + 1
    while period <= size:
        residue_energy = np.bincount(positions % period, weights=edges[positions - 1], minlength=period)
        phase = int(residue_energy.argmax())
        leading_padding = (period - phase) % period
        logical_size = (leading_padding + size + period - 1) // period
        if logical_size <= maximum_logical_size:
            return period, phase
        period += 1
    return size, 0


def blob_periods(blob, maximum_width, maximum_height):
    rgb8 = (blob[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    color_codes = torch.where(blob[..., 3] > 0, color_codes, -1)
    period_x, phase_x = estimate_pixel_period(color_codes, maximum_width)
    period_y, phase_y = estimate_pixel_period(color_codes.transpose(0, 1), maximum_height)
    return period_x, phase_x, period_y, phase_y


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


def collapse_blob(blob, period_x, phase_x, period_y, phase_y):
    height, width, channels = blob.shape
    left = (period_x - phase_x) % period_x
    top = (period_y - phase_y) % period_y
    padded_width = left + width
    padded_height = top + height
    right = (-padded_width) % period_x
    bottom = (-padded_height) % period_y
    aligned = torch.zeros((padded_height + bottom, padded_width + right, channels), dtype=blob.dtype, device=blob.device)
    aligned[top:top + height, left:left + width] = blob

    logical_height = aligned.shape[0] // period_y
    logical_width = aligned.shape[1] // period_x
    cells = aligned.reshape(logical_height, period_y, logical_width, period_x, channels)
    cells = cells.permute(0, 2, 1, 3, 4).reshape(-1, period_y * period_x, channels)
    cell_count = cells.shape[0]
    cell_ids = torch.arange(cell_count, device=blob.device)[:, None].expand_as(cells[..., 0]).reshape(-1)
    center_index = (period_y // 2) * period_x + period_x // 2
    selected = select_modal_rgba(cells.reshape(-1, channels), cell_ids, cell_count, cells[:, center_index])
    logical = selected.reshape(logical_height, logical_width, channels)
    visible = logical[..., 3] > 0
    if not torch.any(visible):
        return None
    rows, columns = torch.where(visible)
    return logical[rows.min():rows.max() + 1, columns.min():columns.max() + 1]


def pad_blob(blob, width, height):
    blob_height, blob_width = blob.shape[:2]
    if blob_width > width or blob_height > height:
        raise ValueError(f"Collapsed blob size {blob_width}x{blob_height} exceeds target size {width}x{height}")
    output = torch.zeros((height, width, 4), dtype=blob.dtype, device=blob.device)
    top = (height - blob_height) // 2
    left = (width - blob_width) // 2
    output[top:top + blob_height, left:left + blob_width] = blob
    return output


def extract_frame_blobs(image, width, height):
    rgb8 = (image[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    color_codes = (rgb8[..., 0] << 16) | (rgb8[..., 1] << 8) | rgb8[..., 2]
    background = color_codes == color_codes[0, 0]
    visible = ~background
    if image.shape[-1] > 3:
        visible &= image[..., 3] > 0

    rgba = torch.zeros((*image.shape[:2], 4), dtype=image.dtype, device=image.device)
    rgba[..., :3] = image[..., :3]
    rgba[..., 3] = visible.to(dtype=image.dtype)
    blobs = []
    for component in visible_components(visible):
        top, bottom = component["top"], component["bottom"]
        left, right = component["left"], component["right"]
        mask = component_mask(component, image.device)
        blob = rgba[top:bottom, left:right].clone()
        blob *= mask[..., None]
        period_x, phase_x, period_y, phase_y = blob_periods(blob, width, height)
        collapsed = collapse_blob(blob, period_x, phase_x, period_y, phase_y)
        if collapsed is not None:
            blobs.append(pad_blob(collapsed, width, height))
    return blobs


def fit_strip_to_frame(strip, width, height):
    scale = min(width / strip.shape[1], height / strip.shape[0])
    scaled_width = max(1, min(width, round(strip.shape[1] * scale)))
    scaled_height = max(1, min(height, round(strip.shape[0] * scale)))
    scaled = torch.nn.functional.interpolate(
        strip.permute(2, 0, 1).unsqueeze(0), size=(scaled_height, scaled_width), mode="nearest-exact"
    )[0].permute(1, 2, 0)
    output = torch.zeros((height, width, 4), dtype=strip.dtype, device=strip.device)
    top = (height - scaled_height) // 2
    left = (width - scaled_width) // 2
    output[top:top + scaled_height, left:left + scaled_width] = scaled
    return output


def composite_white(image):
    alpha = image[..., 3:4].clamp(0.0, 1.0)
    return image[..., :3] * alpha + (1.0 - alpha)


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
                io.Image.Input("palette_image", optional=True,
                               tooltip="Use this image's colors as the palette for the entire batch. Palettes above colors are reduced first. Overrides shared_palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, color_reduction_first, scale_to_original, allow_uneven_grid=False,
                shared_palette=False, palette_image=None, x_offset=0.0, y_offset=0.0, auto_offset=False):
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
            fixed_palette = palette_from_image(palette_image, image[0, 0, 0, :3], colors, image.dtype, image.device)
        detected_offsets = detect_pixel_grid_offsets(image, width, height) if auto_offset else None

        if color_reduction_first:
            reduced = reduce_image_batch(image, colors, shared_palette, fixed_palette)
            return io.NodeOutput(collapse_pixel_grid_mode(reduced, width, height, scale_to_original, allow_uneven_grid,
                                                          x_offset, y_offset, detected_offsets))

        collapsed = collapse_pixel_grid_mode(image, width, height, scale_to_original, allow_uneven_grid, x_offset, y_offset,
                                             detected_offsets)
        return io.NodeOutput(reduce_image_batch(collapsed, colors, shared_palette, fixed_palette))


class MiniMaxH3PixelArtAutorefiner(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3PixelArtAutorefiner",
            display_name="MiniMax H3 Pixel Art Autorefiner",
            description="Finds visible blobs, detects their pixel periods, collapses and packs them into sprite strips.",
            category="image/minimax",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("width", default=64, min=1, max=16384, step=1,
                             tooltip="Width of the transparent canvas allocated to each collapsed blob."),
                io.Int.Input("height", default=64, min=1, max=16384, step=1,
                             tooltip="Height of the transparent canvas allocated to each collapsed blob."),
                io.Int.Input("colors", default=24, min=2, max=256, step=1,
                             tooltip="Maximum generated or supplied palette size."),
                io.Boolean.Input("scale_to_original", default=True,
                                 tooltip="Scale each sprite strip to fit and pad it to the input frame dimensions."),
                io.Boolean.Input("shared_palette", default=True,
                                 tooltip="Generate one palette from the entire image batch instead of a separate palette for each frame."),
                io.Image.Input("palette_image", optional=True,
                               tooltip="Use this image's colors as the palette for the entire batch. Overrides shared_palette."),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, image, width, height, colors, scale_to_original, shared_palette=False, palette_image=None):
        if image.shape[-1] < 3:
            raise ValueError(f"MiniMax H3 pixel art autorefiner requires at least 3 image channels, got {image.shape[-1]}")
        if palette_image is not None and palette_image.shape[-1] < 3:
            raise ValueError(f"Palette image requires at least 3 image channels, got {palette_image.shape[-1]}")

        fixed_palette = None
        if palette_image is not None:
            fixed_palette = palette_from_image(palette_image, image[0, 0, 0, :3], colors, image.dtype, image.device)
        reduced = reduce_image_batch(image, colors, shared_palette, fixed_palette)
        frame_blobs = [extract_frame_blobs(frame, width, height) for frame in reduced]

        if scale_to_original:
            output = []
            for blobs in frame_blobs:
                if blobs:
                    strip = torch.cat(blobs, dim=1)
                else:
                    strip = torch.zeros((height, width, 4), dtype=image.dtype, device=image.device)
                output.append(fit_strip_to_frame(strip, image.shape[2], image.shape[1]))
            return io.NodeOutput(composite_white(torch.stack(output)))

        blob_count = max(1, max((len(blobs) for blobs in frame_blobs), default=0))
        empty_blob = torch.zeros((height, width, 4), dtype=image.dtype, device=image.device)
        output = []
        for blobs in frame_blobs:
            output.append(torch.cat(blobs + [empty_blob] * (blob_count - len(blobs)), dim=1))
        return io.NodeOutput(composite_white(torch.stack(output)))


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
