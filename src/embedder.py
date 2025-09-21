from transformers import AutoModel, AutoProcessor
import torch

class MultiModalEmbedder:
    def __init__(self, device="cuda", model="jinaai/jina-clip-v2"):
        self.device = device
        self.model = AutoModel.from_pretrained(model,
                                               trust_remote_code=True)
        self.model = self.model.to(device).half().eval()
        self.processor = AutoProcessor.from_pretrained(model,
                                                       trust_remote_code=True)

    def embed_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            outputs = self.model.get_text_features(**inputs).cpu().tolist()
        return outputs

    def embed_images(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.get_image_features(**inputs).cpu().tolist()
        return outputs