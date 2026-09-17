# ComfyUI Krea 2 Pixel Art Refiner

A single ComfyUI node that turns Krea 2 output into grid-aligned, limited-palette pixel art.

## Processing

The node performs two operations. By default, it runs them in this order:

1. Selects a perceptual palette with weighted OKLab clustering and maps the image to that palette.
2. Collapses the result to the requested logical pixel grid. Each cell uses its most frequent 8-bit RGB color. Ties are resolved by choosing the color perceptually closest to the cell's center pixel in OKLab; any remaining tie uses the first matching pixel in raster order.

Disable `color_reduction_first` to reverse the order and collapse the grid before palette reduction.

By default the result is expanded back to the input dimensions with nearest-neighbor scaling. Disable `scale_to_original` to return the logical grid at exactly `width × height`. The node supports image batches and alpha channels and uses only PyTorch supplied by ComfyUI.

Input dimensions must normally divide evenly into the logical grid. Enable `allow_uneven_grid` to distribute remainder rows and columns across the grid while preserving the full image and its original dimensions.

Use `x_offset` and `y_offset` to shift the grid phase by a fraction of one logical pixel when generated pixel boundaries do not start at the image edge. For example, `0.5` shifts the grid by half a logical pixel. Offset processing wraps at the outer image edges.

Enable `auto_offset` to detect the strongest repeating X and Y edge phases separately for every image in a batch. Flat margins contribute negligible periodic evidence. Automatic detection overrides the manual offsets and falls back to zero on an image axis with no detectable repeating edges.

Enable `shared_palette` to generate one palette from the entire input batch and apply it to every image. Connect `palette_image` to use its RGB colors as the palette instead; this overrides `shared_palette`. If the palette image contains more unique colors than `colors`, it is perceptually reduced to that limit first. The upper-left RGB color of the first input image is then added so a differing background remains available.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/envy-ai/ComfyUI-Krea2-Pixel-Art-Refiner.git krea2_pixel_art_refiner
```

Restart ComfyUI after installation.

## Node

Find **Krea 2 Pixel Art Refiner** under `image/krea2`.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | ComfyUI image or image batch. |
| `width` | 64 | Logical pixel-grid width. |
| `height` | 64 | Logical pixel-grid height. |
| `colors` | 24 | Maximum colors selected separately for each image. |
| `color_reduction_first` | On | Reduce colors before collapsing the pixel grid. |
| `scale_to_original` | On | Expand the logical grid back to the input dimensions. |
| `allow_uneven_grid` | Off | Permit grid cells of slightly different sizes when the input dimensions do not divide evenly. |
| `shared_palette` | Off | Generate and apply one palette across the entire image batch. |
| `x_offset` | 0.0 | Shift the grid right by a fraction of one logical pixel. |
| `y_offset` | 0.0 | Shift the grid down by a fraction of one logical pixel. |
| `auto_offset` | Off | Detect X/Y grid phase for every image and override the manual offsets. |
| `palette_image` | — | Optional image whose RGB colors become the shared palette, reduced to `colors` when needed, plus the first input image's upper-left color. |

When `allow_uneven_grid` is disabled, the input width and height must be evenly divisible by the requested logical width and height.

Typical wiring:

```text
VAE Decode → Krea 2 Pixel Art Refiner → Save Image
```
