from abc import ABC, abstractmethod
import torch
import utils.output_manager as OutputManager
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.utils as utils
import os
import handlers.prune as pruners
import utils.datasets as datasets

class RetrievalBase(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.device = opt['device']
        # get original model
        self.text_encoder, self.vision_encoder, self.tokenizer, self.preprocessor = utils.get_original_model(opt, self.device) 
        # get run batch function tailored to the model
        self.run_batch = utils.get_run_fn(self.opt['model_info']['model_name']) 

    @torch.no_grad()
    def run(self):
        dataset = self.opt['inference_dataset']['class'](
            dataset_info=self.opt['inference_dataset'], 
            split='test', 
            concept='all', 
            preprocess=self.preprocessor
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.opt['batch_size'],
            num_workers=4,
            shuffle=False,
            pin_memory=True
        )

        output = OutputManager.Output()

        for incremental_id, safe_image, nsfw_image, safe_caption, nsfw_caption, _ in tqdm(dataloader, desc='Inference -- Dataset'):
            text_safe_embeddings, text_nsfw_embeddings, visual_safe_embeddings, visual_nsfw_embeddings = self.run_batch(
                self.tokenizer,
                self.text_encoder,
                self.vision_encoder,
                safe_image,
                nsfw_image,
                safe_caption,
                nsfw_caption,
                self.device
            )

            output.add(
                text_safe_embeddings,
                text_nsfw_embeddings,
                visual_safe_embeddings,
                visual_nsfw_embeddings
            )

        res = output.get_output()
        self.compute_and_save_results(res)
        
    def compute_and_save_results(self, res):
        save_dir = self.opt['save_path']
        os.makedirs(save_dir, exist_ok=True)
        utils.compute_accuracy_and_save(
            self.opt,
            res,
            save_dir
        )

class RetrievalUWM(RetrievalBase):
    def __init__(self, opt):
        super().__init__(opt)
        self.text_pruner = None
        self.vision_pruner = None
        if opt['text_encoder_scorer'] is not None:
            self.text_pruner = pruners.TextPrunerManager(
                text_encoder = self.text_encoder,
                tokenizer = self.tokenizer,
                dataset_info = self.opt['text_encoder_pruning_dataset'],
                opt = self.opt,
                device = self.device
            )
            self.vision_encoder = self.vision_encoder.cpu()
            self.text_pruner.score()
            self.vision_encoder = self.vision_encoder.to(self.device)
        if opt['vision_encoder_scorer'] is not None:
            self.vision_pruner = pruners.VisionPrunerManager(
                vision_encoder = self.vision_encoder,
                preprocessor = self.preprocessor,
                dataset_info = self.opt['vision_encoder_pruning_dataset'],
                opt = self.opt,
                device = self.device
            )
            self.text_encoder = self.text_encoder.cpu()
            self.vision_pruner.score()
            self.text_encoder = self.text_encoder.to(self.device)
        assert self.text_pruner is not None or self.vision_pruner is not None, 'At least one scorer should be initialized'

    def set_mask(self, concept, verbose=False):
        if self.text_pruner is not None: self.text_pruner.set_inference_mask(concept, verbose)
        if self.vision_pruner is not None: self.vision_pruner.set_inference_mask(concept, verbose)

    @torch.no_grad()
    def run(self):
        self.set_mask(concept='all')
        super().run()

class RetrievalInformedPruning(RetrievalBase):
    def __init__(self, opt):
        super().__init__(opt)
        self.text_pruner = None
        self.vision_pruner = None
        if opt['text_encoder_scorer'] is not None:
            self.text_pruner = pruners.GradientTextPrunerManager(
                text_encoder = self.text_encoder,
                tokenizer = self.tokenizer,
                vision_encoder = self.vision_encoder,
                preprocessor = self.preprocessor,
                dataset_info = self.opt['vision_encoder_pruning_dataset'], # we need both modalities
                opt = self.opt,
                device = self.device
            )
            self.text_pruner.score()
        if opt['vision_encoder_scorer'] is not None:
            self.vision_pruner = pruners.GradientVisionPrunerManager(
                vision_encoder = self.vision_encoder,
                preprocessor = self.preprocessor,
                text_encoder = self.text_encoder,
                tokenizer = self.tokenizer,
                dataset_info = self.opt['vision_encoder_pruning_dataset'],
                opt = self.opt,
                device = self.device
            )
            self.vision_pruner.score()
        assert self.text_pruner is not None or self.vision_pruner is not None, 'At least one scorer should be initialized'

    def set_mask(self, concept, verbose=False):
        if self.text_pruner is not None: self.text_pruner.set_inference_mask(concept, verbose)
        if self.vision_pruner is not None: self.vision_pruner.set_inference_mask(concept, verbose)

    @torch.no_grad()
    def run(self):
        self.set_mask(concept='all')
        super().run()