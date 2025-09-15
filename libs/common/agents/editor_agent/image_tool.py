"""Image generation tool for agent framework using typed models and DI.

- Strong typing for inputs/outputs
- No internal LLMService construction; supports DI or backend dependencies
"""

from enum import StrEnum

from agents.tools.base_tool import BaseTool
from core.llm_service import LLMService, ModelSpeed
from core.logging_service import get_logger
from pydantic import BaseModel, Field
from utils.image_utils import ensure_image_directories

logger = get_logger(__name__)


# Typed enums for constrained parameters
class ImageSize(StrEnum):
    SQUARE_1024 = "1024x1024"
    WIDE_1792_1024 = "1792x1024"
    TALL_1024_1792 = "1024x1792"


class ImageQuality(StrEnum):
    STANDARD = "high"  # OpenAI expects 'high' for standard quality
    HD = "hd"


class ImageStyle(StrEnum):
    VIVID = "vivid"
    NATURAL = "natural"


class ImageToolParams(BaseModel):
    """Validated input for image generation."""
    prompt: str = Field(...,
                        description="Text description of the image to generate")
    size: ImageSize = Field(
        default=ImageSize.SQUARE_1024, description="Image size")
    quality: ImageQuality = Field(
        default=ImageQuality.STANDARD, description="Image quality")
    style: ImageStyle = Field(default=ImageStyle.VIVID,
                              description="Image style")
    n: int = Field(default=1, ge=1, le=4,
                   description="Number of images to generate (1-4)")
    story_id: str | None = Field(default=None,
                                description="Optional story ID for filename tracking")


class GeneratedImage(BaseModel):
    url: str
    local_path: str | None = None
    revised_prompt: str | None = None


def _empty_generated_image_list() -> list[GeneratedImage]:
    return []


class ImageToolResult(BaseModel):
    success: bool
    images: list[GeneratedImage] = Field(
        default_factory=_empty_generated_image_list)
    model: str | None = None
    prompt: str | None = None
    error: str | None = None


class ImageGenerationTool(BaseTool):
    # Declare params model at class level to avoid needing an __init__ override
    params_model: type[BaseModel] | None = ImageToolParams

    def __init__(self, llm_service: LLMService | None = None) -> None:
        """Initialize the image generation tool.

        Args:
            llm_service: Injected LLMService instance. Must be set before execution.
        """
        name: str = "generate_image"
        description: str = """
Generate images from a text prompt using AI image generation.

REQUIRED PARAMETERS:
- prompt (required): Text description of the image to generate

OPTIONAL PARAMETERS:
- size: Image dimensions - "1024x1024" (default), "1792x1024", "1024x1792"
- quality: Image quality - "standard" (default) or "hd"
- style: Image style - "vivid" (default) or "natural"
- n: Number of images to generate (1-4, default: 1)

CORRECT USAGE EXAMPLES:
<tool>generate_image</tool><args>{"prompt": "a newspaper on a desk"}</args>
<tool>generate_image</tool><args>{"prompt": "breaking news illustration", "size": "1792x1024", "quality": "hd"}</args>
<tool>generate_image</tool><args>{"prompt": "tech conference", "style": "natural", "n": 2}</args>

INCORRECT USAGE (DO NOT USE):
- {"description": "image text"} ❌ Wrong parameter name
- {"text": "image prompt"} ❌ Wrong parameter name
- {"prompt": ["text1", "text2"]} ❌ Should be single string
- {"size": "large"} ❌ Must use exact dimensions

RETURNS:
{
  "success": bool,
  "images": [{"url": str, "local_path": str, "revised_prompt": str}],
  "model": str,
  "prompt": str,
  "error": str | null
}
"""

        super().__init__(name=name, description=description)
        if llm_service:
            self.set_llm_service(llm_service)

        # Ensure image directories exist
        ensure_image_directories()

    async def execute(self, params: BaseModel, model_speed: ModelSpeed = ModelSpeed.FAST) -> ImageToolResult:
        """Generate an image from a text prompt with strict typing."""
        if not isinstance(params, ImageToolParams):
            return ImageToolResult(success=False, error="Invalid parameters type for ImageGenerationTool")
        if not params.prompt:
            return ImageToolResult(success=False, error="Prompt parameter is required")
        if self.get_llm_service() is None:
            return ImageToolResult(success=False, error="LLMService is required for ImageGenerationTool")

        logger.info(
            "Generating image",
            prompt=params.prompt[:100],
            size=params.size.value,
            n=params.n,
        )

        try:
            response = self.get_llm_service().generate_image(
                prompt=params.prompt,
                size=params.size.value,
                quality=params.quality.value,
                style=params.style.value,
                n=params.n,
            )

            images: list[GeneratedImage] = []
            for image_data in response.data:
                if image_data.url:
                    # Keep the URL for the generated image (no local download)
                    generated_img = GeneratedImage(
                        url=image_data.url,
                        local_path=None,  # No local storage needed
                        revised_prompt=image_data.revised_prompt or params.prompt,
                    )

                    logger.info(
                        "Generated image created",
                        url=image_data.url,
                        prompt=params.prompt[:50],
                        story_id=params.story_id or "unknown"
                    )

                    images.append(generated_img)
                elif image_data.b64_json:
                    # Handle base64 encoded images - convert to data URL
                    data_url = f"data:image/png;base64,{image_data.b64_json}"
                    generated_img = GeneratedImage(
                        url=data_url,
                        local_path=None,
                        revised_prompt=image_data.revised_prompt or params.prompt,
                    )

                    logger.info(
                        "Generated image created (base64)",
                        data_url_length=len(data_url),
                        prompt=params.prompt[:50],
                        story_id=params.story_id or "unknown"
                    )

                    images.append(generated_img)

            return ImageToolResult(
                success=True,
                images=images,
                model=response.model,
                prompt=params.prompt,
            )
        except Exception as e:
            logger.exception("Image generation failed")
            return ImageToolResult(success=False, error=str(e), prompt=params.prompt)
