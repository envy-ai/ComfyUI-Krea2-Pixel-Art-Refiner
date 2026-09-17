# ComfyUI Pixel Art Refiners

Three ComfyUI nodes that turn generated images and video frames into grid-aligned, limited-palette pixel art.

All three nodes select perceptual palettes with weighted OKLab clustering and collapse images to a logical pixel grid. Each grid cell uses its most frequent 8-bit RGB color. Ties are resolved by choosing the color perceptually closest to the cell's center pixel in OKLab; any remaining tie uses the first matching pixel in raster order.

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

## MiniMax H3 Pixel Art Autorefiner

Find **MiniMax H3 Pixel Art Autorefiner** under `image/minimax`. It detects and collapses one pixel mesh across each complete frame:

1. Reduce the frame to the requested perceptual palette.
2. Upscale the source two times and detect edges with Canny and morphological closing.
3. Find horizontal and vertical grid lines with a probabilistic Hough transform, cluster nearby lines, estimate the pixel size from the median filtered gaps, and complete the mesh.
4. Collapse each mesh cell with the same modal-color and center-pixel tie breaking used by the other refiners.
5. Treat the upper-left color as transparent, optionally scale and pad the complete collapsed frame to the original dimensions, then composite it on white.

The mesh detector is adapted from [Proper Pixel Art](https://github.com/KennethJAllen/proper-pixel-art). Its palette quantizer is not used.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | ComfyUI image, video-frame batch, or image batch. |
| `width` | 64 | Fallback logical width if automatic mesh detection fails. |
| `height` | 64 | Fallback logical height if automatic mesh detection fails. |
| `colors` | 24 | Maximum generated or reduced palette size. |
| `scale_to_original` | On | Fit the collapsed frame inside the original frame and pad it to the original dimensions. |
| `shared_palette` | On | Generate one palette across the input batch. |
| `palette_image` | — | Optional image whose colors become the palette for every frame, plus the first input frame's upper-left color. |

For image batches, the node detects the mesh from the first frame and applies that exact mesh to every later frame. This keeps sprite size and position stable across an animation without combining edge evidence from different frames. With `scale_to_original` disabled, the node returns the detected true pixel resolution.

Typical wiring:

```text
VAE Decode → Pixel Art Refiner → Save Image or Create Video
```
