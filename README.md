# ComfyUI Krea 2 Pixel Art Refiner

A single ComfyUI node that turns Krea 2 output into grid-aligned, limited-palette pixel art.

## Processing

The node performs two operations. By default, it runs them in this order:

1. Selects a perceptual palette with weighted OKLab clustering and maps the image to that palette.
2. Collapses the result to the requested logical pixel grid. Each cell uses its most frequent 8-bit RGB color. Ties are resolved by choosing the color perceptually closest to the cell's center pixel in OKLab; any remaining tie uses the first matching pixel in raster order.

Disable `color_reduction_first` to reverse the order and collapse the grid before palette reduction.

The result retains the input dimensions, supports image batches and alpha channels, and uses only PyTorch supplied by ComfyUI.

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

The input width and height must be evenly divisible by the requested logical width and height.

Typical wiring:

```text
VAE Decode → Krea 2 Pixel Art Refiner → Save Image
```
