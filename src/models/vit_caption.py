from transformers import VisionEncoderDecoderModel
from transformers import ViTImageProcessor
from transformers import AutoTokenizer
from PIL import Image
import torch

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

feature_extractor = ViTImageProcessor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")

    pixel_values = feature_extractor(
        images=image,
        return_tensors="pt"
    ).pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        max_length=30,
        num_beams=4
    )

    caption = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption
if __name__ == "__main__":
    image_path = "src/models/abd.jpg"   # adjust path if needed

    caption = generate_caption(image_path)

    print("\nGenerated Caption:")
    print(caption)