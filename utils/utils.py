import random
import torch
import utils.safe_ground_metrics as sgm
import os
from tabulate import tabulate
import handlers.prune as pruners

# import model specific libraries
from transformers import CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPProcessor, AutoProcessor, AutoModel
import open_clip
# captioning metrics
import nltk.translate.meteor_score as meteor_score
from torcheval.metrics.functional.text import bleu_score
from rouge_score import rouge_scorer

def compute_meteor_score(hypothesis, reference):
    return meteor_score.single_meteor_score(reference = reference.split(), hypothesis = hypothesis.split())

def compute_bleu_score(hypothesis, reference, n_gram=4):
    return bleu_score([hypothesis], [reference], n_gram=n_gram)

def compute_rouge_score(hypothesis, reference):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

def compute_cider_score(hypothesis, reference):
    pass

def get_pruners(opt, text_encoder, vision_encoder, tokenizer, preprocessor, device):
    text_pruner = None
    vision_pruner = None
    if opt['text_encoder_scorer'] is not None:
        text_pruner = pruners.TextPrunerManager(
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            dataset_info = opt['text_encoder_pruning_dataset'],
            opt = opt,
            device = device
        )
        if vision_encoder is not None: vision_encoder = vision_encoder.cpu()
        text_pruner.score()
        if vision_encoder is not None: vision_encoder = vision_encoder.to(device)
    if opt['vision_encoder_scorer'] is not None:
        vision_pruner = pruners.VisionPrunerManager(
            vision_encoder = vision_encoder,
            preprocessor = preprocessor,
            dataset_info = opt['vision_encoder_pruning_dataset'],
            opt = opt,
            device = device
        )
        if text_pruner is not None: text_encoder = text_encoder.cpu()
        vision_pruner.score()
        if text_pruner is not None: text_encoder = text_encoder.to(device)
    assert text_pruner is not None or vision_pruner is not None, 'At least one pruner should be initialized'
    return text_pruner, vision_pruner

def get_pruners_gradient(opt, text_encoder, vision_encoder, tokenizer, preprocessor, device):
    text_pruner = None
    vision_pruner = None
    if opt['text_encoder_scorer'] is not None:
        text_pruner = pruners.GradientTextPrunerManager(
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            vision_encoder = vision_encoder,
            preprocessor = preprocessor,
            dataset_info = opt['text_encoder_pruning_dataset'],
            opt = opt,
            device = device
        )
        text_pruner.score()
    if opt['vision_encoder_scorer'] is not None:
        vision_pruner = pruners.GradientVisionPrunerManager(
            vision_encoder = vision_encoder,
            preprocessor = preprocessor,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            dataset_info = opt['vision_encoder_pruning_dataset'],
            opt = opt,
            device = device
        )
        vision_pruner.score()
    assert text_pruner is not None or vision_pruner is not None, 'At least one pruner should be initialized'
    return text_pruner, vision_pruner

def run_batch_clip(
    tokenizer,
    text_encoder,
    vision_encoder,
    safe_image,
    nsfw_image,
    safe_caption,
    nsfw_caption,
    device
):
    safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)
    nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)

    text_safe_embeddings = text_encoder(**safe_ids)
    text_safe_embeddings.text_embeds = text_safe_embeddings.text_embeds.to('cpu')
    text_safe_embeddings.last_hidden_state = text_safe_embeddings.last_hidden_state.to('cpu')
    safe_ids = safe_ids.to('cpu')

    text_nsfw_embeddings = text_encoder(**nsfw_ids)
    text_nsfw_embeddings.text_embeds = text_nsfw_embeddings.text_embeds.to('cpu')
    text_nsfw_embeddings.last_hidden_state = text_nsfw_embeddings.last_hidden_state.to('cpu')
    nsfw_ids = nsfw_ids.to('cpu')

    safe_image = safe_image.to(device)
    nsfw_image = nsfw_image.to(device)
    visual_safe_embeddings = vision_encoder(**{'pixel_values': safe_image})
    visual_safe_embeddings.image_embeds = visual_safe_embeddings.image_embeds.to('cpu')
    visual_safe_embeddings.last_hidden_state = visual_safe_embeddings.last_hidden_state.to('cpu')
    safe_image = safe_image.to('cpu')

    visual_nsfw_embeddings = vision_encoder(**{'pixel_values': nsfw_image})
    visual_nsfw_embeddings.image_embeds = visual_nsfw_embeddings.image_embeds.to('cpu')
    visual_nsfw_embeddings.last_hidden_state = visual_nsfw_embeddings.last_hidden_state.to('cpu')
    nsfw_image = nsfw_image.to('cpu')

    return text_safe_embeddings.text_embeds, text_nsfw_embeddings.text_embeds, visual_safe_embeddings.image_embeds, visual_nsfw_embeddings.image_embeds

def run_batch_siglip(
    tokenizer,
    text_encoder,
    vision_encoder,
    safe_image,
    nsfw_image,
    safe_caption,
    nsfw_caption,
    device
):
    safe_ids = tokenizer(safe_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)
    nsfw_ids = tokenizer(nsfw_caption, return_tensors='pt', padding='max_length', truncation=True).to(device)

    text_safe_embeddings = text_encoder(**safe_ids)
    text_safe_embeddings.pooler_output = text_safe_embeddings.pooler_output.to('cpu')
    text_safe_embeddings.last_hidden_state = text_safe_embeddings.last_hidden_state.to('cpu')
    safe_ids = safe_ids.to('cpu')

    text_nsfw_embeddings = text_encoder(**nsfw_ids)
    text_nsfw_embeddings.pooler_output = text_nsfw_embeddings.pooler_output.to('cpu')
    text_nsfw_embeddings.last_hidden_state = text_nsfw_embeddings.last_hidden_state.to('cpu')
    nsfw_ids = nsfw_ids.to('cpu')

    safe_image = safe_image.to(device)
    nsfw_image = nsfw_image.to(device)
    visual_safe_embeddings = vision_encoder(**{'pixel_values': safe_image})
    visual_safe_embeddings.pooler_output = visual_safe_embeddings.pooler_output.to('cpu')
    visual_safe_embeddings.last_hidden_state = visual_safe_embeddings.last_hidden_state.to('cpu')
    safe_image = safe_image.to('cpu')

    visual_nsfw_embeddings = vision_encoder(**{'pixel_values': nsfw_image})
    visual_nsfw_embeddings.pooler_output = visual_nsfw_embeddings.pooler_output.to('cpu')
    visual_nsfw_embeddings.last_hidden_state = visual_nsfw_embeddings.last_hidden_state.to('cpu')
    nsfw_image = nsfw_image.to('cpu')

    return text_safe_embeddings.pooler_output, text_nsfw_embeddings.pooler_output, visual_safe_embeddings.pooler_output, visual_nsfw_embeddings.pooler_output

def run_batch_coca(
    tokenizer,
    text_encoder,
    vision_encoder,
    safe_image,
    nsfw_image,
    safe_caption,
    nsfw_caption,
    device
):
    safe_ids = tokenizer(safe_caption).to(device)
    nsfw_ids = tokenizer(nsfw_caption).to(device)

    text_safe_embeddings = text_encoder(safe_ids)
    text_safe_embeddings = (text_safe_embeddings[0].to('cpu'), text_safe_embeddings[1].to('cpu'))
    safe_ids = safe_ids.to('cpu')

    text_nsfw_embeddings = text_encoder(nsfw_ids)
    text_nsfw_embeddings = (text_nsfw_embeddings[0].to('cpu'), text_nsfw_embeddings[1].to('cpu'))
    nsfw_ids = nsfw_ids.to('cpu')

    safe_image = safe_image.to(device)
    nsfw_image = nsfw_image.to(device)
    visual_safe_embeddings = vision_encoder(safe_image)
    visual_safe_embeddings = (visual_safe_embeddings[0].to('cpu'), visual_safe_embeddings[1].to('cpu'))
    safe_image = safe_image.to('cpu')

    visual_nsfw_embeddings = vision_encoder(nsfw_image)
    visual_nsfw_embeddings = (visual_nsfw_embeddings[0].to('cpu'), visual_nsfw_embeddings[1].to('cpu'))
    nsfw_image = nsfw_image.to('cpu')

    return text_safe_embeddings[0], text_nsfw_embeddings[0], visual_safe_embeddings[0], visual_nsfw_embeddings[0]

def get_run_fn(model_name):
    if 'coca' in model_name:
        return run_batch_coca
    elif 'siglip' in model_name:
        return run_batch_siglip
    elif 'clip' in model_name:
        return run_batch_clip
    else:
        raise NotImplementedError(f"Model {model_name} not Supported")
    
def set_deterministic(seed):
    # set all seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def get_original_model(opt, device):
    if 'siglip' in opt['model_info']['model_name']:
        print('Loading SigLIP model...')
        model = AutoModel.from_pretrained(opt['model_info']['model_name'])
        processor = AutoProcessor.from_pretrained(opt['model_info']['model_name'])
        text_encoder = model.text_model.to(device)
        vision_encoder = model.vision_model.to(device)
        tokenizer = processor.tokenizer
        preprocessor = processor.image_processor
    elif 'coca' in opt['model_info']['model_name']:
        print('Loading COCA model...')
        model, _, transform = open_clip.create_model_and_transforms(
            model_name=opt['model_info']['model_name'],
            pretrained=opt['model_info']['pretrained'],
        )
        text_encoder = model.text.to(device)
        vision_encoder = model.visual.to(device)
        tokenizer = open_clip.get_tokenizer(opt['model_info']['model_name'])
        preprocessor = transform
    else:
        print('Loading CLIP model...')
        text_encoder = CLIPTextModelWithProjection.from_pretrained(opt['model_info']['model_name']).to(device)
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained(opt['model_info']['model_name']).to(device)
        tokenizer = CLIPTokenizer.from_pretrained(opt['model_info']['tokenizer_name'])
        preprocessor = CLIPProcessor.from_pretrained(opt['model_info']['preprocessor'])
    print('Model Loaded')
    return text_encoder, vision_encoder, tokenizer, preprocessor

def is_linear(layer):
    return isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.modules.linear.NonDynamicallyQuantizableLinear)

def recall(temb, vemb, K=(1,5,10,20)):
    num_text = temb.shape[0]
    num_im = vemb.shape[0]
    text_to_image_map = image_to_text_map = torch.LongTensor(tuple(i for i in range(num_text)))

    # text-to-image recall
    dist_matrix = temb.cpu() @ vemb.cpu().T  # dist_matrix[i] gives logits for ith text

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    text_to_image_recall = []
    for k in K:
        # Extract top k indices only
        topk = inds[:, :k]
        # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)

    # image-to-text recall
    dist_matrix = dist_matrix.T  # dist_matrix[i] gives logits for the ith image
    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    image_to_text_recall = []
    for k in K:
        # Extract top k indices only
        topk = inds[:, :k]
        correct = torch.eq(topk, image_to_text_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_im)#

    text_to_image_recall = [str(round(i*100, 2)) for i in text_to_image_recall]
    image_to_text_recall = [str(round(i*100, 2)) for i in image_to_text_recall]

    return text_to_image_recall, image_to_text_recall

def recall_union(safe_text, unsafe_text, safe_image, unsafe_image, K=(1,5,10,20)):
    # unsafe text vs U(safe image, unsafe image) setting
    num_text = unsafe_text.shape[0]
    text_to_image_map = image_to_text_map = torch.LongTensor(tuple(i for i in range(num_text)))
    dist_matrix_safe = unsafe_text.cpu() @ safe_image.cpu().T
    dist_matrix_unsafe = unsafe_text.cpu() @ unsafe_image.cpu().T
    ovearall_dist_matrix = torch.cat((dist_matrix_safe, dist_matrix_unsafe), dim=1)

    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(ovearall_dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    text_to_image_recall = []
    for k in K:
        topk = inds[:, :k]
        correct = torch.eq(topk, text_to_image_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        text_to_image_recall.append(num_correct / num_text)
    
    # unsafe image vs U(safe text, unsafe text) setting
    dist_matrix_safe = unsafe_image.cpu() @ safe_text.cpu().T
    dist_matrix_unsafe = unsafe_image.cpu() @ unsafe_text.cpu().T
    ovearall_dist_matrix = torch.cat((dist_matrix_safe, dist_matrix_unsafe), dim=1)
    # Sort in descending order; first is the biggest logit
    inds = torch.argsort(ovearall_dist_matrix, dim=1, descending=True)
    inds = inds.to(text_to_image_map.device)

    image_to_text_recall = []
    for k in K:
        topk = inds[:, :k]
        correct = torch.eq(topk, image_to_text_map.unsqueeze(-1)).any(dim=1)
        num_correct = correct.sum().item()
        image_to_text_recall.append(num_correct / num_text)

    text_to_image_recall = [str(round(i*100, 2)) for i in text_to_image_recall]
    image_to_text_recall = [str(round(i*100, 2)) for i in image_to_text_recall]

    return text_to_image_recall, image_to_text_recall

def compute_recall(all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings):
    K=(1,5,10,20)
    
    all_text_safe_embeddings = all_text_safe_embeddings / all_text_safe_embeddings.norm(dim=-1, keepdim=True)
    all_text_nsfw_embeddings = all_text_nsfw_embeddings / all_text_nsfw_embeddings.norm(dim=-1, keepdim=True)
    all_visual_safe_embeddings = all_visual_safe_embeddings / all_visual_safe_embeddings.norm(dim=-1,keepdim=True)
    all_visual_nsfw_embeddings = all_visual_nsfw_embeddings / all_visual_nsfw_embeddings.norm(dim=-1,keepdim=True)

    safe_setting = recall(all_text_safe_embeddings, all_visual_safe_embeddings, K=K)
    union_ranks = recall_union(all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings, K=K)

    preferences = sgm.compute_preference(all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings)
    safe_ground_scores = sgm.compute_safe_ground(all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings)

    return safe_setting, union_ranks, preferences, safe_ground_scores

def save_results(
    opt,
    safe_setting,
    union_ranks,
    preferences,
    safe_ground_scores,
    save_dir
):
    text_to_image_recall, image_to_text_recall = safe_setting
    text_to_image_recall_union, image_to_text_recall_union = union_ranks
    safe_text_preference, unsafe_text_preference, safe_image_preference, unsafe_image_preference = preferences
    text_score, image_score, safe_score, unsafe_score, group_score = safe_ground_scores

    header_top = ['', '', 'Text-to-Image', '', '', '', 'Image-to-Text', '', '', '', 'UNSAFE Text-to-Union', '', '', '', 'UNSAFE Image-to-Union', '', '', '', 'Preference Metrics', '', '', '', '', 'Safe Ground Metrics', '', '']
    header_bottom = ['Model'] + [f'R@{k}' for k in (1, 5, 10, 20)] * 4 + ['Safe Text', 'Unsafe Text', 'Safe Image', 'Unsafe Image'] + ['Text Score', 'Image Score', 'Safe Score', 'Unsafe Score', 'Group Score']
    data = [opt['model_info']['model_name']] + text_to_image_recall + image_to_text_recall + text_to_image_recall_union + image_to_text_recall_union + \
        [safe_text_preference, unsafe_text_preference, safe_image_preference, unsafe_image_preference] \
        + [text_score, image_score, safe_score, unsafe_score, group_score] 

    table = [header_top, header_bottom, data]
    with open(os.path.join(save_dir, 'results.txt'), "w") as f:
        f.write(tabulate(table, headers="firstrow", tablefmt="grid"))
    with open(os.path.join(save_dir, 'results.tex'), "w") as f:
        f.write(tabulate(table, headers="firstrow", tablefmt="latex"))

def compute_accuracy_and_save(opt, res, save_dir):
    print(f"------------------- Computing accuracy -------------------")
    all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings = res

    safe_setting, union_ranks, preferences, safe_ground_scores = compute_recall(
        all_text_safe_embeddings, all_text_nsfw_embeddings, all_visual_safe_embeddings, all_visual_nsfw_embeddings
    )

    # Safe_Text_Safe_Image
    text_to_image_recall, image_to_text_recall = safe_setting
    print("Safe setting:")
    print("Safe Text to Safe Images recall: ", text_to_image_recall)
    print("Safe Image to Safe Texts recall: ", image_to_text_recall)

    # Union Setting
    text_to_image_recall_union, image_to_text_recall_union = union_ranks
    print("Unsafe union setting:")
    print("Unsafe Text to Union(safe_images, unsafe_images) recall: ", text_to_image_recall_union)
    print("Unsafe Image to Union(safe_texts, unsafe_texts) recall: ", image_to_text_recall_union)

    # Preferences
    safe_text_preference, unsafe_text_preference, safe_image_preference, unsafe_image_preference = preferences
    print("Preferences:")
    print("Safe Text as Query (P^t_s): ", safe_text_preference)
    print("Unsafe Text as Query (P^t_u): ", unsafe_text_preference)
    print("Safe Image as Query (P^v_s): ", safe_image_preference)
    print("Unsafe Image as query (P^v_u): ", unsafe_image_preference)

    text_score, image_score, safe_score, unsafe_score, group_score = safe_ground_scores
    print("Safe Ground Scores")
    print(f"Text Score (Txt_s): {text_score}")
    print(f"Image Score (Img_s): {image_score}")
    print(f"Safe Input Score (PS): {safe_score}")
    print(f"Unsafe Input Score (PU): {unsafe_score}")
    print(f"Group Score (GS): {group_score}")


    print("------------------- END RESULTS -------------------")
    save_results(
        opt,
        safe_setting,
        union_ranks,
        preferences,
        safe_ground_scores,
        save_dir
    )

def fill_template(labels, template):
    prompts = []
    for label in labels:
        prompts.append(template.format(label.replace('_', ' ')))
    return prompts