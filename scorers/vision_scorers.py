import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
import utils.utils as utils
from layers.linear import CustomLinear
import os
import torch.nn.functional as F

class Base(ABC):
    def __init__(
        self,
        vision_encoder,
        concepts,
        layers=None,
        sparsity=0.98,
        alpha=-1,
        coca=False,
        path_scores=None
    ):
        self.vision_encoder = vision_encoder
        self.vision_encoder.eval()
        self.device = next(vision_encoder.parameters()).device
        self.concepts = concepts
        self.sparsity = sparsity
        self.alpha = alpha
        self.coca = coca
        self.save_path_scores = os.path.join(path_scores, 'vision')
        self.vision_tower = vision_encoder.__class__.__name__ == 'CLIPVisionTower'
        self.layers = {}
        self.hooks = []
        self.concept_mask = {}
        self.running_scores = {}
        # counts = 0
        for module in self.vision_encoder.modules():
            for name, layer in module.named_children():
                # if utils.is_linear(layer) and (layers is None or name in layers):
                #     counts += 1
                if utils.is_linear(layer) and (layers is None or name in layers):
                    custom_layer = CustomLinear.from_pretrained(module.__str__(), name, layer)
                    self.hooks.append(
                        custom_layer.register_forward_hook(self.hook_fn)
                    )
                    setattr(module, name, custom_layer)
                    self.layers[id(custom_layer)] = custom_layer
                    self.running_scores[id(custom_layer)] = {
                        concept: torch.zeros_like(custom_layer.weight, device='cpu') for concept in concepts
                    }
                    self.concept_mask[id(custom_layer)] = {
                        concept: torch.ones_like(custom_layer.weight, device='cpu') for concept in concepts
                    }
        assert len(self.hooks) > 0, 'No hooks found, please check the layers to prune'

    def load_masks(self, concept):
        print(f'Loading scores for {concept}...')
        if not os.path.exists(os.path.join(self.save_path_scores, concept)):
            return False
        counts = 0
        for layer in self.layers.values():
            layer_name = str(counts) + '_' + layer.name
            if not os.path.exists(os.path.join(self.save_path_scores, concept, f'{layer_name}.pt')):
                return False
            scores = torch.load(os.path.join(self.save_path_scores, concept, f'{layer_name}.pt'), map_location=torch.device("cpu"), weights_only=True)
            mask = self.compute_mask(scores)
            self.concept_mask[id(layer)][concept] = mask.cpu()
            counts += 1
        print('Done')
        return True
    
    def compute_overall_mask(self, verbose):
        if verbose: print('Computing overall mask...')
        progress_bar = tqdm(self.layers.values(), position=0, leave=True) if verbose else self.layers.values()
        for layer in progress_bar:
            mask = torch.ones_like(layer.weight).to(layer.weight.device)
            for concept in self.concepts:
                mask *= self.concept_mask[id(layer)][concept].to(mask.device)
            self.concept_mask[id(layer)]['all'] = mask.clone()
            layer.update_mask(torch.where(mask == 0, self.alpha, 1))
        if verbose: print('Done')

    def set_inference_mask(self, concept, verbose=True):
        assert concept in self.concepts + ['all'], 'Concept mask not found'
        if concept == 'all':
            self.compute_overall_mask(verbose)
        else:
            if verbose: print(f'Computing {concept} mask...')
            progress_bar = tqdm(self.layers.values(), position=0, leave=True) if verbose else self.layers.values()
            for layer in progress_bar:
                final_mask = torch.where(self.concept_mask[id(layer)][concept] == 0, self.alpha, 1)
                layer.update_mask(final_mask)
            if verbose: print('Done')

    def set_concept(self, concept):
        self.concept = concept

    def valid_tensor(self, tensor):
        return torch.isnan(tensor).sum() == 0 and torch.isinf(tensor).sum() == 0 and torch.isneginf(tensor).sum() == 0

    def hook_fn(self, module, input, output):
        scores = self.scoring_function(module, input[0], output)
        assert self.valid_tensor(scores), 'Invalid tensor'
        self.running_scores[id(module)][self.concept] += scores.cpu()

    def prune(self, n_batches):
        counts = 0
        for layer in self.layers.values():
            scores = self.running_scores[id(layer)][self.concept]/n_batches
            layer_name = str(counts) + '_' + layer.name
            self.save_scores(scores, self.concept, layer_name)
            self.concept_mask[id(layer)][self.concept] = self.compute_mask(scores)
            counts += 1

    def save_scores(self, scores, concept, layer_name):
        os.makedirs(os.path.join(self.save_path_scores, concept), exist_ok=True)
        torch.save(scores, os.path.join(self.save_path_scores, concept, f'{layer_name}.pt'))

    def compute_mask(self, scores):
        k = int(self.sparsity*scores.numel())
        if not k < 1:
            thr, _ = torch.kthvalue(torch.flatten(scores), k)
            return torch.where(scores > thr, 0, 1)
        return torch.ones_like(scores)

    @abstractmethod
    def scoring_function(self, module, input, output):
        pass

    def get_local_splits(self, data):
        assert data.shape[0]%2 == 0, 'The number of prompts should be even for safe and unsafe splits'
        n_prompts = data.shape[0]//2
        safe_activations = data[:n_prompts]
        unsafe_activations = data[n_prompts:]
        return safe_activations, unsafe_activations

    @torch.no_grad()
    def __call__(self, data, concept):
        self.set_concept(concept)
        if not self.coca and not self.vision_tower:
            _ = self.vision_encoder(**{'pixel_values': data})
        else:
            _ = self.vision_encoder(data)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

class UWM(Base):
    # Adaptive Thresholding
    # Eq. 11 of the paper
    def compute_mask(self, score):
        score = score.to(torch.float32)
        assert self.valid_tensor(torch.flatten(score).sum()), 'Sum exceeding floating point precision, please check the scoring function'
        score_norm = torch.flatten(score)/torch.flatten(score).sum()
        values, indices = torch.sort(score_norm)
        y = torch.cumsum(values, dim=0)
        indices_to_prune = indices[y>self.sparsity]
        mask = torch.ones_like(score)
        final_mask = mask.flatten().scatter_(0, indices_to_prune, 0).reshape(score.shape)
        return final_mask

    def saliency_score(self, module, data):
        assert len(data.shape) == 2, f'Invalid shape {data.shape}'
        data_unsqueezed = data.unsqueeze(1)
        scores = (data_unsqueezed * module.weight.abs())
        assert len(scores.shape) == 3, f'Invalid shape {scores.shape}'
        assert scores.shape[1] == module.weight.shape[0], f'Invalid shape [1] {scores.shape}'
        assert scores.shape[0] == data.shape[0], f'Invalid shape [0] {scores.shape}'
        input_sal = scores.mean(dim=1)
        output_sal = scores.mean(dim=2) 
        return torch.einsum('bi,bj->bij', output_sal, input_sal)

    def image_level_norm(self, data):
        norms = []
        stds = []
        for idx, image_act in enumerate(data):
            norms.append(torch.norm(image_act, p=2, dim=0, keepdim=True))
            stds.append(image_act.std(dim=0).unsqueeze(0))
        return torch.cat(norms), torch.cat(stds)

    def score(self, module, data):
        assert len(data.shape) == 3, f'Invalid data shape {data.shape}'
        assert len(data.shape) == 3 and data.shape[1] > 1, f'Invalid data [1] shape {data.shape}'
        safe_activations, unsafe_activations = self.get_local_splits(data)

        # Image level norms
        safe_norms, safe_stds = self.image_level_norm(safe_activations)
        unsafe_norms, unsafe_stds = self.image_level_norm(unsafe_activations)
        
        # Eq. 8 of the paper
        safe_saliency = self.saliency_score(module, safe_norms)
        unsafe_saliency = self.saliency_score(module, unsafe_norms)

        # Eq. 9 of the paper
        safe_score = safe_saliency/(safe_stds+1e-8).unsqueeze(1)
        unsafe_score = unsafe_saliency/(unsafe_stds+1e-8).unsqueeze(1)

        # Eq. 10 of the paper -- Final score (i.e., ratio of unsafe and safe scores)
        final_score = (unsafe_score/safe_score).mean(dim=0)
        return final_score

    def scoring_function(self, module, input, output):
        score = self.score(module, input).to(module.weight.device)
        return score

# GRAD baselines
class InformedPruning():
    def __init__(
        self,
        vision_encoder,
        text_encoder,
        concepts,
        layers=None,
        sparsity=0.98,
        alpha=-1,
        coca=False,
        path_scores=None
    ):
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_encoder.eval()
        self.device = next(vision_encoder.parameters()).device
        self.concepts = concepts
        self.sparsity = sparsity
        self.alpha = alpha
        self.coca = coca
        self.save_path_scores = os.path.join(path_scores, 'vision')
        self.layers = {}
        self.backward_hooks = []
        self.concept_mask = {}
        self.running_grads = {}
        for module in self.vision_encoder.modules():
            for name, layer in module.named_children():
                if utils.is_linear(layer) and (layers is None or name in layers):
                    custom_layer = CustomLinear.from_pretrained(module.__str__(), name, layer)
                    custom_layer.weight.requires_grad = True
                    self.backward_hooks.append(
                        custom_layer.weight.register_post_accumulate_grad_hook(self.backward_hook_fn)
                    )
                    self.layers[id(custom_layer)] = custom_layer
                    self.concept_mask[id(custom_layer)] = {
                        concept: torch.ones_like(custom_layer.weight, device='cpu') for concept in concepts
                    }
                    self.running_grads[id(custom_layer.weight)] = {
                        concept: torch.zeros_like(custom_layer.weight, device='cpu') for concept in concepts
                    }
                    setattr(module, name, custom_layer)
    
    def load_masks(self, concept):
        print(f'Loading masks for {concept}...')
        if not os.path.exists(os.path.join(self.save_path_scores, concept)):
            return False
        counts = 0
        for layer in self.layers.values():
            layer_name = str(counts) + '_' + layer.name
            if not os.path.exists(os.path.join(self.save_path_scores, concept, f'{layer_name}.pt')):
                return False
            scores = torch.load(os.path.join(self.save_path_scores, concept, f'{layer_name}.pt'), map_location=torch.device("cpu"), weights_only=True)
            mask = self.compute_score(layer, scores)
            self.concept_mask[id(layer)][concept] = mask.cpu()
            counts += 1
        print('Done')
        return True
    
    def compute_overall_mask(self, verbose):
        if verbose: print('Computing overall mask...')
        progress_bar = tqdm(self.layers.values(), position=0, leave=True) if verbose else self.layers.values()
        for layer in progress_bar:
            mask = torch.ones_like(layer.weight).to(layer.weight.device)
            for concept in self.concepts:
                mask *= self.concept_mask[id(layer)][concept].to(mask.device)
            self.concept_mask[id(layer)]['all'] = mask.clone()
            layer.update_mask(torch.where(mask == 0, self.alpha, 1))
        if verbose: print('Done')

    def set_inference_mask(self, concept, verbose=True):
        assert concept in self.concepts + ['all'], 'Concept mask not found'
        if concept == 'all':
            self.compute_overall_mask(verbose)
        else:
            if verbose: print(f'Computing {concept} mask...')
            progress_bar = tqdm(self.layers.values(), position=0, leave=True) if verbose else self.layers.values()
            for layer in progress_bar:
                final_mask = torch.where(self.concept_mask[id(layer)][concept] == 0, self.alpha, 1)
                layer.update_mask(final_mask)
            if verbose: print('Done')

    def set_concept(self, concept):
        self.concept = concept
    
    def scoring_fn(self, module, mean_grads):
        return mean_grads

    def valid_tensor(self, tensor):
        return torch.isnan(tensor).sum() == 0 and torch.isinf(tensor).sum() == 0 and torch.isneginf(tensor).sum() == 0

    def backward_hook_fn(self, parameters):
        assert parameters.grad is not None, 'Gradient is None'
        grads = parameters.grad.abs().detach().clone().cpu()
        assert grads.sum() != 0, 'Gradient is zero'
        assert self.valid_tensor(grads), 'Invalid tensor'
        self.running_grads[id(parameters)][self.concept] += grads

    def image_forward(self, image):
        image = image.to(self.device, non_blocking=True)
        if not self.coca:
            output = self.vision_encoder(**{'pixel_values': image})
            if hasattr(output, 'image_embeds'):
                image_features = output.image_embeds
            elif hasattr(output, 'pooler_output'):
                image_features = output.pooler_output
            else:
                raise NotImplementedError('Image encoder output not supported')
        else:
            image_features = self.vision_encoder(image)[0]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def text_forward(self, text):
        if not self.coca:   
            text = {k: v.to(self.device, non_blocking=True) for k, v in text.items()}
            output = self.text_encoder(**text)
            if hasattr(output, 'text_embeds'): # CLIP
                text_features = output.text_embeds
            elif hasattr(output, 'pooler_output'): # Siglip
                text_features = output.pooler_output
            else:
                raise NotImplementedError('Text encoder output not supported')
        else:
            text = text.to(self.device, non_blocking=True)
            text_features = self.text_encoder(text)[0]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def loss_fn(self, image_features, text_features):
        # maximaze cosine similarity between safe text and unsafe image
        cos_sim = (image_features*text_features).sum(dim=-1)
        return 1-cos_sim.mean(dim=0)

    def compute_mask(self, score):
        score_norm = torch.flatten(score)/torch.flatten(score).sum()
        values, indices = torch.sort(score_norm)
        y = torch.cumsum(values, dim=0)
        indices_to_prune = indices[y>self.sparsity]
        mask = torch.ones_like(score)
        final_mask = mask.flatten().scatter_(0, indices_to_prune, 0).reshape(score.shape)
        return final_mask

    def __call__(self, safe_image, unsafe_image, text_unsafe, text_safe, current_concept):
        self.set_concept(current_concept)
        with torch.no_grad():
            unsafe_text_features = self.text_forward(text_unsafe)
        safe_image_features = self.image_forward(safe_image)
        loss = self.loss_fn(safe_image_features, unsafe_text_features)
        loss.backward()

    def compute_score(self, module, mean_grads):
        scoring = self.scoring_fn(module, mean_grads)
        self.valid_tensor(scoring)
        return self.compute_mask(scoring)

    def prune(self, n_batches):
        counts = 0
        for layer in tqdm(self.layers.values(), position=int(self.device.index)+1, leave=False, desc='Layers'):
            grads = self.running_grads[id(layer.weight)][self.concept]
            assert grads.sum() != 0, f'No gradients were accumulated for the concept {self.concept}.'
            mean_grads = grads/n_batches
            assert self.valid_tensor(mean_grads), 'Mean grads are invalid'
            layer_name = str(counts) + '_' + layer.name
            self.save_scores(mean_grads, self.concept, layer_name)
            mask = self.compute_score(layer, mean_grads)
            self.concept_mask[id(layer)][self.concept] = mask
            counts += 1

    def save_scores(self, scores, concept, layer_name):
        os.makedirs(os.path.join(self.save_path_scores, concept), exist_ok=True)
        torch.save(scores, os.path.join(self.save_path_scores, concept, f'{layer_name}.pt'))

    def remove_hooks(self):
        for hook in self.backward_hooks:
            hook.remove()

# Gradient Safe CLIP pruning baseline
class GradientSafeCLIP(InformedPruning):
    def redirection_loss(self, text_features, image_features):
        labels = torch.arange(text_features.shape[0], device=text_features.device)
        temperature = 1.
        logit_scale = (torch.ones([]) * 4.6052).exp()

        # Cosine similarity as logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # Classic CLIP loss
        loss_clip = 0.5 * (
            F.cross_entropy(logits_per_image / temperature, labels)
            + F.cross_entropy(logits_per_text / temperature, labels)
        )
        
        return loss_clip
    
    def distance_loss(self, text_features, image_features):
        return 1-torch.nn.CosineSimilarity()(text_features, image_features)

    def __call__(self, safe_image, unsafe_image, text_unsafe, text_safe, current_concept):
        self.vision_encoder.zero_grad()
        self.set_concept(current_concept)
        with torch.no_grad():
            unsafe_text_features = self.text_forward(text_unsafe)
            unsafe_image_features = self.image_forward(unsafe_image)
        safe_image_features = self.image_forward(safe_image)
        loss = self.redirection_loss(unsafe_text_features, safe_image_features)
        loss += self.distance_loss(safe_image_features, unsafe_image_features).mean(dim=0)
        loss.backward()
    
    def compute_mask(self, scores):
        k = int(self.sparsity*scores.numel())
        if not k < 1:
            thr, _ = torch.kthvalue(torch.flatten(scores), k)
            return torch.where(scores > thr, 0, 1)
        return torch.ones_like(scores)

# Gradient Unsafe pruning baseline
class GradientUnsafe(InformedPruning):
    def __call__(self, safe_image, unsafe_image, text_unsafe, text_safe, current_concept):
        self.vision_encoder.zero_grad()
        self.set_concept(current_concept)
        with torch.no_grad():
            unsafe_text_features = self.text_forward(text_unsafe)
            safe_text_features = self.text_forward(text_safe)
        safe_image_features = self.image_forward(safe_image)
        loss = - self.loss_fn(safe_image_features, safe_text_features) + self.loss_fn(safe_image_features, unsafe_text_features)
        loss.backward()


# Ablation studies below

class Ablation_UWM_Unsafe_Only(UWM):
    # hard thresholding
    def compute_mask(self, scores):
        k = int(self.sparsity*scores.numel())
        if not k < 1:
            thr, _ = torch.kthvalue(torch.flatten(scores), k)
            return torch.where(scores > thr, 0, 1)
        return torch.ones_like(scores)

    def score(self, module, data):
        assert len(data.shape) == 3, f'Invalid data shape {data.shape}'
        assert len(data.shape) == 3 and data.shape[1] > 1, f'Invalid data [1] shape {data.shape}'
        safe_activations, unsafe_activations = self.get_local_splits(data)

        # Image level norms
        # safe_norms, safe_stds = self.image_level_norm(safe_activations)
        unsafe_norms, unsafe_stds = self.image_level_norm(unsafe_activations)

        # safe_saliency = self.saliency_score(module, safe_norms)
        unsafe_saliency = self.saliency_score(module, unsafe_norms)

        # safe_score = safe_saliency/(safe_stds+1e-8).unsqueeze(1)
        unsafe_score = unsafe_saliency/(unsafe_stds+1e-8).unsqueeze(1)

        # Final score (i.e., ratio of unsafe and safe scores)
        final_score = unsafe_score.mean(dim=0)
        return final_score

class Supp_Ablation_UWM_Unsafe_Only_Adaptive(Ablation_UWM_Unsafe_Only):
    # Adaptive Thresholding
    def compute_mask(self, score):
        score = score.to(torch.float32)
        assert self.valid_tensor(torch.flatten(score).sum()), 'Sum exceeding floating point precision, please check the scoring function'
        score_norm = torch.flatten(score)/torch.flatten(score).sum()
        values, indices = torch.sort(score_norm)
        y = torch.cumsum(values, dim=0)
        indices_to_prune = indices[y>self.sparsity]
        mask = torch.ones_like(score)
        final_mask = mask.flatten().scatter_(0, indices_to_prune, 0).reshape(score.shape)
        return final_mask

class Supp_Ablation_UWM_Unsafe_Saliency_Only_Adaptive(UWM):
    # Adaptive Thresholding
    def compute_mask(self, score):
        score = score.to(torch.float32)
        assert self.valid_tensor(torch.flatten(score).sum()), 'Sum exceeding floating point precision, please check the scoring function'
        score_norm = torch.flatten(score)/torch.flatten(score).sum()
        values, indices = torch.sort(score_norm)
        y = torch.cumsum(values, dim=0)
        indices_to_prune = indices[y>self.sparsity]
        mask = torch.ones_like(score)
        final_mask = mask.flatten().scatter_(0, indices_to_prune, 0).reshape(score.shape)
        return final_mask

    def score(self, module, data):
        assert len(data.shape) == 3, f'Invalid data shape {data.shape}'
        assert len(data.shape) == 3 and data.shape[1] > 1, f'Invalid data [1] shape {data.shape}'
        safe_activations, unsafe_activations = self.get_local_splits(data)

        # Image level norms
        # safe_norms, safe_stds = self.image_level_norm(safe_activations)
        unsafe_norms, unsafe_stds = self.image_level_norm(unsafe_activations)

        # safe_saliency = self.saliency_score(module, safe_norms)
        unsafe_saliency = self.saliency_score(module, unsafe_norms)

        # safe_score = safe_saliency/(safe_stds+1e-8).unsqueeze(1)
        unsafe_score = unsafe_saliency

        # Final score (i.e., ratio of unsafe and safe scores)
        final_score = unsafe_score.mean(dim=0)
        return final_score

class Supp_Ablation_UWM_Diff(UWM):
    # hard thresholding
    def compute_mask(self, scores):
        k = int(self.sparsity*scores.numel())
        if not k < 1:
            thr, _ = torch.kthvalue(torch.flatten(scores), k)
            return torch.where(scores > thr, 0, 1)
        return torch.ones_like(scores)

    def score(self, module, data):
        assert len(data.shape) == 3, f'Invalid data shape {data.shape}'
        assert len(data.shape) == 3 and data.shape[1] > 1, f'Invalid data [1] shape {data.shape}'
        safe_activations, unsafe_activations = self.get_local_splits(data)

        # Image level norms
        safe_norms, safe_stds = self.image_level_norm(safe_activations)
        unsafe_norms, unsafe_stds = self.image_level_norm(unsafe_activations)

        safe_saliency = self.saliency_score(module, safe_norms)
        unsafe_saliency = self.saliency_score(module, unsafe_norms)

        safe_score = safe_saliency/(safe_stds+1e-8).unsqueeze(1)
        unsafe_score = unsafe_saliency/(unsafe_stds+1e-8).unsqueeze(1)

        # Final score (i.e., ratio of unsafe and safe scores)
        final_score = (unsafe_score - safe_score).mean(dim=0)
        return final_score

class Supp_Ablation_UWM_Diff_Adaptive(Supp_Ablation_UWM_Diff):
    # Adaptive Thresholding
    def compute_mask(self, score):
        score = score.to(torch.float32)
        assert self.valid_tensor(torch.flatten(score).sum()), 'Sum exceeding floating point precision, please check the scoring function'
        score_norm = torch.flatten(score)/torch.flatten(score).sum()
        values, indices = torch.sort(score_norm)
        y = torch.cumsum(values, dim=0)
        indices_to_prune = indices[y>self.sparsity]
        mask = torch.ones_like(score)
        final_mask = mask.flatten().scatter_(0, indices_to_prune, 0).reshape(score.shape)
        return final_mask

class Ablation_UWM_Div(Ablation_UWM_Unsafe_Only):
    def score(self, module, data):
        assert len(data.shape) == 3, f'Invalid data shape {data.shape}'
        assert len(data.shape) == 3 and data.shape[1] > 1, f'Invalid data [1] shape {data.shape}'
        safe_activations, unsafe_activations = self.get_local_splits(data)

        # Image level norms
        safe_norms, safe_stds = self.image_level_norm(safe_activations)
        unsafe_norms, unsafe_stds = self.image_level_norm(unsafe_activations)

        safe_saliency = self.saliency_score(module, safe_norms)
        unsafe_saliency = self.saliency_score(module, unsafe_norms)

        safe_score = safe_saliency/(safe_stds+1e-8).unsqueeze(1)
        unsafe_score = unsafe_saliency/(unsafe_stds+1e-8).unsqueeze(1)

        # Final score (i.e., ratio of unsafe and safe scores)
        final_score = (unsafe_score/safe_score).mean(dim=0)
        return final_score

