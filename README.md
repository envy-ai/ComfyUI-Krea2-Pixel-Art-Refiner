# ComfyUI Pixel Art Refiners

Two ComfyUI nodes that turn generated images and video frames into grid-aligned, limited-palette pixel art.

Both nodes select perceptual palettes with weighted OKLab clustering and collapse images to a logical pixel grid. Each grid cell uses its most frequent 8-bit RGB color. Ties are resolved by choosing the color perceptually closest to the cell's center pixel in OKLab; any remaining tie uses the first matching pixel in raster order.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/envy-ai/ComfyUI-Krea2-Pixel-Art-Refiner.git krea2_pixel_art_refiner
```

Restart ComfyUI after installation.

## Krea 2 Pixel Art Refiner

Find **Krea 2 Pixel Art Refiner** under `image/krea2`. This is the original focused node with palette reduction, grid collapse, selectable processing order, and optional scaling back to the input dimensions.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | ComfyUI image or image batch. |
| `width` | 64 | Logical pixel-grid width. |
| `height` | 64 | Logical pixel-grid height. |
| `colors` | 24 | Maximum colors selected separately for each image. |
| `color_reduction_first` | On | Reduce colors before collapsing the pixel grid. |
| `scale_to_original` | On | Expand the logical grid back to the input dimensions. |

The input width and height must be evenly divisible by the requested logical width and height.

## MiniMax H3 Pixel Art Refiner

Find **MiniMax H3 Pixel Art Refiner** under `image/minimax`. It includes the full batch and video-oriented feature set.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | ComfyUI image, video-frame batch, or image batch. |
| `width` | 64 | Logical pixel-grid width. |
| `height` | 64 | Logical pixel-grid height. |
| `colors` | 24 | Maximum generated or reduced palette size. |
| `color_reduction_first` | On | Reduce colors before collapsing the pixel grid. |
| `scale_to_original` | On | Expand the logical grid back to the input dimensions. |
| `allow_uneven_grid` | Off | Permit grid cells of slightly different sizes when dimensions do not divide evenly. |
| `shared_palette` | Off | Generate and apply one palette across the entire image batch. |
| `x_offset` | 0.0 | Shift the grid right by a fraction of one logical pixel. |
| `y_offset` | 0.0 | Shift the grid down by a fraction of one logical pixel. |
| `auto_offset` | Off | Detect X/Y grid phase for every image and override the manual offsets. |
| `palette_image` | — | Optional image whose RGB colors become the shared palette, reduced to `colors` when needed, plus the first input image's upper-left color. |

Manual offsets range from `0.0` to `1.0`; for example, `0.5` shifts the grid by half a logical pixel. Offset processing wraps at the outer image edges.

Automatic offset detection finds the strongest repeating X and Y edge phases separately for every image. Flat margins contribute negligible periodic evidence. An axis with no detectable repetition falls back to zero.

When a palette image contains no more than `colors` unique RGB values, those exact colors are retained. Larger palettes are perceptually reduced first. The first input image's upper-left RGB color is then added so a differing background remains available.

Typical wiring:

```text
VAE Decode → Pixel Art Refiner → Save Image or Create Video
```
