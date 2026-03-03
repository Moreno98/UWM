import torch
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.datasets import ConceptDataset
import utils.utils as utils

class TextPrunerManager(ABC):
    def __init__(
        self,
        text_encoder,
        tokenizer,
        dataset_info,
        opt,
        device
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dataset_info = dataset_info
        self.opt = opt
        self.device = device
        self.coca = False

        if 'coca' in opt['model_name'].lower():
            self.coca = True
            self.tokenize = self.tokenize_coca

        self.concepts = opt['concepts']
        self.dataset_size = opt['pruning_dataset_size']
        self.text_encoder_scorer = opt['text_encoder_scorer']['class'](
            text_encoder = text_encoder,
            concepts = self.concepts,
            layers = opt['text_encoder_layers'],
            sparsity = opt['sparsity_text'],
            alpha = opt['alpha_text'],
            coca = self.coca,
            path_scores = opt['path_scores']
        )

        fc_layer = 'fc' in opt['text_encoder_layers'][0] 

        self.batch_size = 2048
        if 'coca' in opt['model_name'].lower() or 'siglip' in opt['model_name'].lower() or fc_layer:
            self.batch_size = 512
        self.shuffle = False
        if self.concepts == ['all']:
            self.shuffle = True
    
    def tokenize(self, forward_data):
        forward_data = self.tokenizer(forward_data, return_tensors='pt', padding='max_length', truncation=True).to(self.device)
        if 'attention_mask' not in forward_data:
            att_mask = torch.where(forward_data['input_ids'] == 1, 0., 1.)
        else:
            att_mask = forward_data['attention_mask']
        return forward_data, att_mask

    def tokenize_coca(self, forward_data):
        forward_data = self.tokenizer(forward_data).to(self.device)
        att_mask = torch.where(forward_data == 0, 0., 1.)
        return forward_data, att_mask

    def set_inference_mask(self, concept, verbose=False):
        self.text_encoder_scorer.set_inference_mask(concept, verbose)

    def score(self):
        print('Scoring text encoder...')
        # utils.set_deterministic(self.opt['seed'])
        textual_dataset = self.dataset_info['class'](
            dataset_info=self.dataset_info, 
            split='train', 
            concept='all',
            subset=self.dataset_size,
            shuffle=self.shuffle
        )
        print('Dataset size:', len(textual_dataset))

        if len(self.concepts) > 1:
            progress_bar = tqdm(self.concepts, position=0, leave=True, desc='Concepts')
            pos = 1
        else:
            progress_bar = self.concepts
            pos = 0

        for concept in progress_bar:
            loaded_masks = self.text_encoder_scorer.load_masks(concept)
            if not loaded_masks:
                print(f'Scores not found for concept: {concept}, scoring...')
                concept_data = textual_dataset.get_concept_data(concept)
                data_loader = torch.utils.data.DataLoader(
                    concept_data,
                    batch_size=self.batch_size,
                    num_workers=1,
                    shuffle=False,
                    pin_memory=True
                )
                n_batches = len(data_loader)

                for incremental_id, safe_text, unsafe_text, tag in tqdm(data_loader, position=pos, leave=False, desc='Batches'):
                    forward_data = safe_text + unsafe_text
                    forward_data, att_mask = self.tokenize(forward_data)
                    self.text_encoder_scorer(
                        data = forward_data,
                        att_mask = att_mask,
                        concept = concept
                    )
                self.text_encoder_scorer.prune(n_batches)
        self.text_encoder_scorer.remove_hooks()
        print('Text encoder scoring completed.')

class VisionPrunerManager():
    def __init__(
        self,
        vision_encoder,
        preprocessor,
        dataset_info,
        opt,
        device
    ):
        self.vision_encoder = vision_encoder
        self.preprocessor = preprocessor
        self.dataset_info = dataset_info
        self.opt = opt
        self.device = device
        self.concepts = opt['concepts']
        self.dataset_size = opt['pruning_dataset_size']
        self.coca = True if 'coca' in opt['model_name'].lower() else False

        self.vision_encoder_scorer = opt['vision_encoder_scorer']['class'](
            vision_encoder = vision_encoder,
            concepts = self.concepts,
            layers = opt['vision_encoder_layers'],
            sparsity = opt['sparsity_vision'],
            alpha = opt['alpha_vision'],
            coca = self.coca,
            path_scores = opt['path_scores']
        )
        
        fc_layer = 'fc' in opt['vision_encoder_layers'][0] 

        self.batch_size = 256
        if 'coca' in opt['model_name'].lower() or 'siglip' in opt['model_name'].lower() or fc_layer:
            self.batch_size = 128
        self.shuffle = False
        if self.concepts == ['all']:
            self.shuffle = True

    def set_inference_mask(self, concept, verbose=False):
        self.vision_encoder_scorer.set_inference_mask(concept, verbose)

    def score(self):
        print('Scoring vision encoder...')
        # utils.set_deterministic(self.opt['seed'])
        visual_dataset = self.dataset_info['class'](
            dataset_info=self.dataset_info, 
            split='train', 
            concept='all',
            preprocess=self.preprocessor,
            subset=self.dataset_size,
            shuffle=self.shuffle
        )
        print('Dataset size:', len(visual_dataset))
        
        if len(self.concepts) > 1:
            progress_bar = tqdm(self.concepts, position=0, leave=True, desc='Concepts')
            pos = 1
        else:
            progress_bar = self.concepts
            pos = 0

        for concept in progress_bar:
            loaded_masks = self.vision_encoder_scorer.load_masks(concept)
            if not loaded_masks:
                print(f'Masks not found for concept: {concept}, scoring...')
                concept_data = visual_dataset.get_concept_data(concept)
                concept_dataset = ConceptDataset(
                    data = concept_data,
                    preprocess = self.preprocessor,
                )
                data_loader = torch.utils.data.DataLoader(
                    concept_dataset,
                    batch_size=self.batch_size,
                    num_workers=4,
                    shuffle=False,
                    pin_memory=True
                )
                n_batches = len(data_loader)

                for incremental_id, safe_image, nsfw_image, safe_text, unsafe_text, tag in tqdm(data_loader, position=pos, leave=False, desc='Batches'):
                    forward_data = torch.cat([safe_image, nsfw_image]).to(self.device)
                    self.vision_encoder_scorer(
                        data = forward_data,
                        concept = concept
                    )
                self.vision_encoder_scorer.prune(n_batches)
        self.vision_encoder_scorer.remove_hooks()
        print('Vision encoder scoring completed.')

#### GRADIENT BASED PRUNING ####
class GradientTextPrunerManager(TextPrunerManager):
    def __init__(
        self,
        text_encoder,
        tokenizer,
        vision_encoder,
        preprocessor,
        dataset_info,
        opt,
        device
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder
        self.preprocessor = preprocessor
        self.dataset_info = dataset_info
        self.opt = opt
        self.device = device
        self.coca = False

        if 'coca' in opt['model_name'].lower():
            self.coca = True
            self.tokenize = self.tokenize_coca

        self.concepts = opt['concepts']
        self.dataset_size = opt['pruning_dataset_size']
        self.text_encoder_scorer = opt['text_encoder_scorer']['class'](
            text_encoder = text_encoder,
            vision_encoder = vision_encoder,
            concepts = self.concepts,
            layers = opt['text_encoder_layers'],
            sparsity = opt['sparsity_text'],
            alpha = opt['alpha_text'],
            coca = self.coca,
            path_scores = opt['path_scores']
        )
        self.batch_size = 512
        if 'coca' in opt['model_name'].lower() or 'siglip' in opt['model_name'].lower():
            self.batch_size = 128
        self.shuffle = False
        if self.concepts == ['all']:
            self.shuffle = True

    def score(self):
        print('Scoring Text encoder...')
        dataset = self.dataset_info['class'](
            dataset_info=self.dataset_info, 
            split='train', 
            concept='all',
            preprocess=self.preprocessor,
            subset=self.dataset_size,
            shuffle=self.shuffle
        )
        if len(self.concepts) > 1:
            progress_bar = tqdm(self.concepts, position=0, leave=True, desc='Concepts')
            pos = 1
        else:
            progress_bar = self.concepts
            pos = 0

        for concept in progress_bar:
            loaded_masks = self.text_encoder_scorer.load_masks(concept)
            if not loaded_masks:
                print(f'Masks not found for concept: {concept}, scoring...')
                concept_data = dataset.get_concept_data(concept)
                concept_dataset = ConceptDataset(
                    data = concept_data,
                    preprocess = self.preprocessor,
                )
                data_loader = torch.utils.data.DataLoader(
                    concept_dataset,
                    batch_size=self.batch_size,
                    num_workers=4,
                    shuffle=False,
                    pin_memory=True
                )
                n_batches = len(data_loader)

                for incremental_id, safe_image, nsfw_image, safe_text, unsafe_text, tag in tqdm(data_loader, position=pos, leave=False, desc='Batches'):
                    safe_text_tokens, _ = self.tokenize(safe_text)
                    unsafe_text_tokens, _ = self.tokenize(unsafe_text)
                    self.text_encoder_scorer(
                        safe_text = safe_text_tokens,
                        unsafe_text = unsafe_text_tokens,
                        image_unsafe = nsfw_image,
                        image_safe = safe_image,
                        current_concept = concept
                    )
                self.text_encoder_scorer.prune(n_batches)
        self.text_encoder_scorer.remove_hooks()
        print('Text encoder pruning completed.')

class GradientVisionPrunerManager(VisionPrunerManager):
    def __init__(
        self,
        vision_encoder,
        preprocessor,
        text_encoder,
        tokenizer,
        dataset_info,
        opt,
        device
    ):
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder
        self.preprocessor = preprocessor
        self.dataset_info = dataset_info
        self.opt = opt
        self.device = device
        self.coca = False

        if 'coca' in opt['model_name'].lower():
            self.coca = True
            self.tokenize = self.tokenize_coca

        self.concepts = opt['concepts']
        self.dataset_size = opt['pruning_dataset_size']
        self.vision_encoder_scorer = opt['vision_encoder_scorer']['class'](
            vision_encoder = vision_encoder,
            text_encoder = text_encoder,
            concepts = self.concepts,
            layers = opt['vision_encoder_layers'],
            sparsity = opt['sparsity_vision'],
            alpha = opt['alpha_vision'],
            coca = self.coca,
            path_scores = opt['path_scores']
        )
        self.batch_size = 64
        if 'coca' in opt['model_name'].lower():
            self.batch_size = 32
        if 'siglip' in opt['model_name'].lower():
            self.batch_size = 16
        self.shuffle = False
        if self.concepts == ['all']:
            self.shuffle = True

    def tokenize_coca(self, forward_data):
        forward_data = self.tokenizer(forward_data).to(self.device)
        att_mask = torch.where(forward_data == 0, 0., 1.)
        return forward_data, att_mask
    
    def tokenize(self, forward_data):
        forward_data = self.tokenizer(forward_data, return_tensors='pt', padding='max_length', truncation=True).to(self.device)
        if 'attention_mask' not in forward_data:
            att_mask = torch.where(forward_data['input_ids'] == 1, 0., 1.)
        else:
            att_mask = forward_data['attention_mask']
        return forward_data, att_mask
    
    def score(self):
        print('Scoring vision encoder...')
        dataset = self.dataset_info['class'](
            dataset_info=self.dataset_info, 
            split='train', 
            concept='all',
            preprocess=self.preprocessor,
            subset=self.dataset_size,
            shuffle=self.shuffle
        )
        if len(self.concepts) > 1:
            progress_bar = tqdm(self.concepts, position=0, leave=True, desc='Concepts')
            pos = 1
        else:
            progress_bar = self.concepts
            pos = 0

        for concept in progress_bar:
            loaded_masks = self.vision_encoder_scorer.load_masks(concept)
            if not loaded_masks:
                print(f'Masks not found for concept: {concept}, scoring...')
                concept_data = dataset.get_concept_data(concept)
                concept_dataset = ConceptDataset(
                    data = concept_data,
                    preprocess = self.preprocessor,
                )
                data_loader = torch.utils.data.DataLoader(
                    concept_dataset,
                    batch_size=self.batch_size,
                    num_workers=4,
                    shuffle=False,
                    pin_memory=True
                )
                n_batches = len(data_loader)

                for incremental_id, safe_image, nsfw_image, safe_text, unsafe_text, tag in tqdm(data_loader, position=pos, leave=False, desc='Batches'):
                    unsafe_text_tokens, _ = self.tokenize(unsafe_text)
                    safe_text_tokens, _ = self.tokenize(safe_text)
                    self.vision_encoder_scorer(
                        safe_image = safe_image,
                        unsafe_image = nsfw_image,
                        text_unsafe = unsafe_text_tokens,
                        text_safe = safe_text_tokens,
                        current_concept = concept
                    )
                self.vision_encoder_scorer.prune(n_batches)
        self.vision_encoder_scorer.remove_hooks()
        print('Vision encoder pruning completed.')