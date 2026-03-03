import utils.arg_parse as arg_parse
from torch.utils.data import DataLoader
import utils.utils as utils
import torch
from tqdm import tqdm
from tabulate import tabulate
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, Lambda
from transformers import AutoProcessor

def main(opt):
    def custom_transform(image):
        if not isinstance(image, torch.Tensor):
            image = image.convert('RGB')
        return preprocessor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)

    def custom_transform_coca(image):
        return preprocessor(image)
    
    device = opt['device']

    # get original model
    text_encoder, vision_encoder, tokenizer, preprocessor = utils.get_original_model(opt, device) 

    if opt['mode'] == 'prune':
        t_pruner, v_pruner = opt['get_pruners_fn'](opt, text_encoder, vision_encoder, tokenizer, preprocessor, device)
        if t_pruner is not None: t_pruner.set_inference_mask('all')
        if v_pruner is not None: v_pruner.set_inference_mask('all')

    coca = 'coca' in opt['model_info']['model_name']

    if coca and opt['inference_dataset']['name'] == 'UCF101':
        preprocessor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14').image_processor
        # preprocessor.do_rescale = False
        # preprocessor = Compose([
        #     Lambda(lambda x: x.permute(2, 0, 1)),
        #     Lambda(lambda x: x.to(torch.float32)),
        #     Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        #     CenterCrop(size=(224, 224)),
        #     Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        # ])
        custom_transform_coca = custom_transform

    dataset = opt['inference_dataset']['get_fn'](
        root = opt['inference_dataset']['root'],
        split = 'test',
        transform = custom_transform if not coca else custom_transform_coca,
        download = True
    )
    dataloader = DataLoader(
        dataset,
        batch_size = opt['batch_size'],
        num_workers = 8,
        shuffle = False,
        pin_memory = True
    )
    prompts = utils.fill_template(opt['inference_dataset']['label_names'], template = opt['inference_dataset']['template'])
    if not coca:
        tokenized_text = tokenizer(prompts, return_tensors='pt', padding='max_length', truncation=True).to(device)
    else:
        tokenized_text = tokenizer(prompts).to(device)
    corrects = 0
    for images, labels in tqdm(dataloader, position=0, leave=True, desc='Inference'):
        images = images.to(device)
        with torch.no_grad():
            if not coca:
                text_output = text_encoder(**tokenized_text)
                image_output = vision_encoder(**{'pixel_values': images})
                if hasattr(text_output, 'text_embeds'):
                    text_features = text_output.text_embeds
                    image_features = image_output.image_embeds
                elif hasattr(text_output, 'pooler_output'):
                    text_features = text_output.pooler_output
                    image_features = image_output.pooler_output
                else:
                    raise NotImplementedError('Model not supported')
            else:
                text_features = text_encoder(tokenized_text)[0]
                image_features = vision_encoder(images)[0]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            preds = similarity.topk(1).indices.squeeze().cpu()
            corrects += (preds == labels).sum().item()

    print(f'Accuracy: {corrects / len(dataset)}')

    headers = ['Model', 'Accuracy']
    table = [
        [opt['model_info']['model_name'], corrects / len(dataset)]
    ]
    with open(f'{opt["save_path"]}/results.txt', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt='grid'))

    with open(f'{opt["save_path"]}/results.tex', 'w') as f:
        f.write(tabulate(table, headers=headers, tablefmt='latex'))

if __name__ == '__main__':
    opt = arg_parse.zero_shot()
    main(opt)