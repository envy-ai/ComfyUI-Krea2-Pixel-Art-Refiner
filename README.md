# ComfyUI Pixel Art Refiners

Four ComfyUI nodes that refine generated pixel art and consolidate noisy held animation frames.

The three refiner nodes select perceptual palettes with weighted OKLab clustering and collapse images to a logical pixel grid. Each grid cell uses its most frequent 8-bit RGB color. Ties are resolved by choosing the color perceptually closest to the cell's center pixel in OKLab; any remaining tie uses the first matching pixel in raster order.

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
| `align_palette_lightness` | On | Estimate and compensate for a shared OKLab lightness shift before matching a supplied palette. |
| `preserve_generated_background` | On | Keep the first input image's upper-left RGB color instead of forcing it into the supplied palette. |
| `palette_image` | — | Optional image whose RGB colors become the shared palette, reduced to `colors` when needed. |

Manual offsets range from `0.0` to `1.0`; for example, `0.5` shifts the grid by half a logical pixel. Offset processing wraps at the outer image edges.

Automatic offset detection finds the strongest repeating X and Y edge phases separately for every image. Flat margins contribute negligible periodic evidence. An axis with no detectable repetition falls back to zero.

When a palette image contains no more than `colors` unique RGB values, those exact colors are retained. Larger palettes are perceptually reduced first. Lightness alignment finds one OKLab L offset for the complete input batch, applies it only while matching colors, and emits exact RGB values from the supplied palette. Using one offset across the batch avoids per-frame color flicker. When generated-background preservation is enabled, that color is excluded from offset estimation and retained exactly; disabling it restricts the entire output to the supplied palette.

## MiniMax H3 Pixel Art Autorefiner

Find **MiniMax H3 Pixel Art Autorefiner** under `image/minimax`. It detects and collapses a pixel mesh across each complete frame, or one mesh inside each fixed horizontal direction region:

1. Reduce the frame to the requested perceptual palette.
2. Unless manual resolution is enabled, divide the frame into `direction_count` equal horizontal regions, upscale each region two times, and detect edges with Canny and morphological closing.
3. Find horizontal and vertical grid lines with a probabilistic Hough transform and cluster nearby lines. Estimate the fundamental pixel size that makes the observed gaps integer multiples, so missing grid landmarks do not turn two or three pixels into one oversized cell. The pitch implied by the requested resolution is preferred when its median and clipped mean landmark errors are both within 20%, making the resolution a hint while still allowing clearly larger source pixels. Two consecutive sub-pitch cells whose combined size is approximately one requested pixel are joined first by removing their shared boundary. Other short false cells are then merged with the neighboring interval that best fits the detected period, and the remaining gaps are subdivided locally so variations in pixel pitch are retained. The requested resolution also sets the smallest allowed pitch and maximum detected grid size.
4. Collapse each mesh cell with the same modal-color and center-pixel tie breaking used by the other refiners.
5. Treat the upper-left color as transparent, optionally scale and pad the complete collapsed frame to the original dimensions, then composite it on white.

The mesh detector is adapted from [Proper Pixel Art](https://github.com/KennethJAllen/proper-pixel-art). Its palette quantizer is not used.

| Input | Default | Description |
| --- | ---: | --- |
| `image` | — | ComfyUI image, video-frame batch, or image batch. |
| `width` | 64 | Full-frame logical width in manual mode, or fallback width if detection fails. |
| `height` | 64 | Full-frame logical height in manual mode, or fallback height if detection fails. |
| `colors` | 24 | Maximum generated or reduced palette size. |
| `scale_to_original` | On | Fit the collapsed frame inside the original frame and pad it to the original dimensions. |
| `shared_palette` | On | Generate one palette across the input batch. |
| `manual_resolution` | Off | Divide the full frame evenly into `width × height` cells instead of detecting a mesh. |
| `direction_count` | 1 | Number of fixed horizontal regions whose X and Y grids are detected independently. Use 4 for a four-direction sprite sheet. |
| `align_palette_lightness` | On | Estimate and compensate for a shared OKLab lightness shift before matching a supplied palette. |
| `preserve_generated_background` | On | Keep the first input frame's upper-left RGB color instead of forcing it into the supplied palette. |
| `palette_image` | — | Optional image whose colors become the palette for every frame. |

For image batches, the node detects every region's mesh from the first frame and applies those exact meshes and fixed region bounds to every later frame. This keeps sprite size and position stable across an animation without combining edge evidence from different frames or directions. `width` remains the logical width of the complete frame; its fallback cells are divided proportionally among the regions. With `scale_to_original` disabled, the independently collapsed regions are padded to a common height and concatenated.

Manual resolution bypasses edge and period detection completely. For example, a `2048 × 512` four-view sheet with `16 × 16` source pixels should use `width=128` and `height=32`.

## Pixel Art Animation Pose Compositor

Find **Pixel Art Animation Pose Compositor** under `image/animation`. It turns a noisy video-frame batch containing repeated animation cycles into one clean image per distinct pose.

The node reproduces the process used on the original 56-frame test:

1. Round every input RGB channel to its 8-bit value.
2. Measure the mean absolute RGB difference between each frame and the next over the complete image.
3. Start a new held pose whenever that difference reaches `transition_threshold`. The default of `7.0` separated the test's within-pose differences (`0.044–1.978`) from its real transitions (`13.182–29.786`).
4. If a single transition frame splits two strongly matching pieces of the same hold, discard that frame and join the pieces.
5. Optionally remove boundary and interior frames whose RGB distance from the hold's medoid exceeds the hold median by the larger of three median absolute deviations or one 8-bit RGB level. The limit is recalculated after each pass and every hold retains at least one frame.
6. Assign detected holds to poses in cycle order. With six poses, holds 1, 7, 13, and so on belong to pose 1; holds 2, 8, 14, and so on belong to pose 2.
7. At every pixel in each pose, select the exact RGB value occurring most often across all frames in its matching holds.
8. If multiple RGB values have the same maximum count, choose the one with the lowest Rec.709 luma. If luma is also tied, choose the lowest packed RGB value for deterministic output.
9. Starting from all image borders, flood through pixels whose RMS RGB distance from the upper-left background color is less than 10%, and replace that connected region with the exact background color. Enclosed near-background pixels are preserved.

| Input | Default | Description |
| --- | ---: | --- |
| `images` | — | Ordered animation frames at a common resolution. |
| `pose_count` | 0 | Autodetect the number of distinct poses from repeated cycles. Set a positive value to specify it explicitly. |
| `transition_threshold` | 7.0 | Minimum full-frame mean absolute RGB difference, measured in 8-bit levels, that begins a new held pose. |
| `remove_outliers` | On | Remove anomalous frames from each hold before cycle detection and compositing. |

With `pose_count=0`, the node selects one medoid frame from every hold and tests possible cycle lengths. It accepts the shortest period where every hold in later cycles is at least twice as close to its corresponding first-cycle pose as it is to any other first-cycle pose. Autodetection therefore requires at least two complete cycles. It can skip an unmatched fragment before those cycles. Any trailing partial cycle is checked against the same phase order; for example, a thirteenth hold after two six-pose cycles is grouped with pose 1. If no unambiguous period is found, the node asks for an explicit pose count instead of combining unrelated poses.

The input must contain at least one complete cycle. Trailing partial cycles are accepted and their holds contribute to the corresponding poses from the start of the cycle. The output is an IMAGE batch of composited frames in animation order.

Typical wiring:

```text
VAE Decode → Pixel Art Refiner → Save Image or Create Video
VAE Decode → Pixel Art Refiner → Animation Pose Compositor → Create Video
```
