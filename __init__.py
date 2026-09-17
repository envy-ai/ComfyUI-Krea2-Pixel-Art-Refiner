from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import Krea2PixelArtRefiner, MiniMaxH3PixelArtAutorefiner, MiniMaxH3PixelArtRefiner


class Krea2PixelArtRefinerExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Krea2PixelArtRefiner, MiniMaxH3PixelArtRefiner, MiniMaxH3PixelArtAutorefiner]


async def comfy_entrypoint() -> Krea2PixelArtRefinerExtension:
    return Krea2PixelArtRefinerExtension()
