import torch

class Output():
    def __init__(self):
        self.all_text_safe_embeddings,\
        self.all_text_nsfw_embeddings,\
        self.all_visual_safe_embeddings,\
        self.all_visual_nsfw_embeddings = [], [], [], []

    def add(
        self,
        text_safe_embeddings,
        text_nsfw_embeddings,
        visual_safe_embeddings,
        visual_nsfw_embeddings
    ):
        self.all_text_safe_embeddings.append(text_safe_embeddings)
        self.all_text_nsfw_embeddings.append(text_nsfw_embeddings)
        self.all_visual_safe_embeddings.append(visual_safe_embeddings)
        self.all_visual_nsfw_embeddings.append(visual_nsfw_embeddings)

    def get_output(self):
        all_text_safe_embeddings = torch.cat(self.all_text_safe_embeddings, 0)
        all_text_nsfw_embeddings = torch.cat(self.all_text_nsfw_embeddings, 0)
        all_visual_safe_embeddings = torch.cat(self.all_visual_safe_embeddings, 0)
        all_visual_nsfw_embeddings = torch.cat(self.all_visual_nsfw_embeddings, 0)

        return all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings
        