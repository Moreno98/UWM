from argparse import RawTextHelpFormatter
import argparse
import utils.config as config
import utils.utils as utils
import os
import handlers.retrieval as retrieval_handlers
import scorers.vision_scorers as vision_scorers
import scorers.text_scorers as text_scorers
import inspect
import handlers.prune as pruners

SCORES_PATH = 'results/scores_path'

def retrieval():
    parser = argparse.ArgumentParser(description='Retrieval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model_name', type=str, default='ViT-B/32', choices=list(config.VLM_MODELS.keys()), help='Model name')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'prune'], help='Mode to use')
    parser.add_argument('--v_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(vision_scorers)).keys()), help='Vision scorer to use')
    parser.add_argument('--t_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(text_scorers)).keys()), help='Scorer to use')
    parser.add_argument('--vision_encoder_pruning_dataset', type=str, choices=list(config.VISION_PRUNING_DTS.keys()), default='ViSU', help='Textual dataset to use while pruning vision encoder')
    parser.add_argument('--text_encoder_pruning_dataset', type=str, choices=list(config.TEXT_PRUNING_DTS.keys()), default='ViSU', help='Textual dataset to use while pruning text encoder')
    parser.add_argument('--sparsity_vision', type=float, default=0.98, help='Sparsity for vision encoder')
    parser.add_argument('--sparsity_text', type=float, default=0.98, help='Sparsity for text encoder')
    parser.add_argument('--alpha_vision', type=float, default=-1, help='Alpha for vision encoder')
    parser.add_argument('--alpha_text', type=float, default=-1, help='Alpha for text encoder')
    parser.add_argument('--text_encoder_layers', nargs='+', default='fc2', help='Layers to prune')
    parser.add_argument('--vision_encoder_layers', nargs='+', default='out_proj', help='Layers to prune')
    parser.add_argument('--inference_dataset', type=str, choices=list(config.DATASETS.keys()), help='Inference dataset to use')
    # make --concept a list
    parser.add_argument('--plot', action='store_true', help='Plots shared pruned weights among layers')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
    parser.add_argument('--pruning_dataset_size', type=int, default=2048, help='Pruning dataset size')
    opt = vars(parser.parse_args())

    opt['inference_dataset'] = config.DATASETS[opt['inference_dataset']]
    seed_str = f'seed_{opt["seed"]}' if opt["seed"] != 0 else ''
    base_save_dir = os.path.join('results', opt['inference_dataset']['name'], 'CLIP_retrieval', str(opt['pruning_dataset_size']), seed_str, opt['model_name'])
    if opt['mode'] == 'original':
        opt['save_path'] = os.path.join(base_save_dir, 'original')
        opt['retrieval_handler'] = retrieval_handlers.RetrievalBase
    else:
        assert set(opt['vision_encoder_layers']).issubset(config.VLM_MODELS[opt['model_name']]['layers']), 'Invalid vision encoder layers, please select from: ' + ', '.join(config.VLM_MODELS[opt['model_name']]['layers'])
        assert set(opt['text_encoder_layers']).issubset(config.VLM_MODELS[opt['model_name']]['layers']), 'Invalid text encoder layers, please select from: ' + ', '.join(config.VLM_MODELS[opt['model_name']]['layers'])
        opt['vision_encoder_scorer'] = {
            'name': opt['v_scorer'],
            'class': getattr(vision_scorers, opt['v_scorer']),
        }
        opt['vision_encoder_pruning_dataset'] = config.VISION_PRUNING_DTS[opt['vision_encoder_pruning_dataset']]
        opt['text_encoder_pruning_dataset'] = config.TEXT_PRUNING_DTS[opt['text_encoder_pruning_dataset']]
        opt['text_encoder_scorer'] = {
            'name': opt['t_scorer'],
            'class': getattr(text_scorers, opt['t_scorer']),
        }
        text_encoder_layers = '' if opt['text_encoder_layers'] is None else 'T_layers_' + '_'.join(opt['text_encoder_layers'])
        vision_encoder_layers = '' if opt['vision_encoder_layers'] is None else 'V_layers_' + '_'.join(opt['vision_encoder_layers'])

        opt['pruning_dataset_size'] = opt['pruning_dataset_size'] if opt['pruning_dataset_size'] != 0 else 2048

        if 'gradient' in opt['vision_encoder_scorer']['name'].lower() or 'gradient' in opt['text_encoder_scorer']['name'].lower():
            opt['concepts'] = ['all']
            concept_mode = 'gradient'
            opt['pruning_dataset_size'] = opt['pruning_dataset_size']*4
            opt['retrieval_handler'] = retrieval_handlers.RetrievalInformedPruning
        else:
            concept_mode = 'UWM'
            opt['concepts'] = ['all']
            # ~400 datapoints per concept
            opt['pruning_dataset_size'] = opt['pruning_dataset_size']*4
            opt['retrieval_handler'] = retrieval_handlers.RetrievalUWM        

        # if alpha == 1 the original encoder is used
        # change the save path accordingly
        if float(opt["alpha_text"]) != 1.0:
            text_encoder_pruning_info = os.path.join(
                'T_ENCODER_' + opt['text_encoder_scorer']['name'],
                text_encoder_layers,
                f'sparsity_{opt["sparsity_text"]}_alpha_{opt["alpha_text"]}',
            )
            text_encoder_scores_path = os.path.join(
                'T_ENCODER_' + opt['text_encoder_scorer']['name'],
                text_encoder_layers
            )
        else:
            single_encoder_pruning = 'VISION_ENCODER_ONLY'
            opt['text_encoder_scorer'] = None
            single_encoder_pruning_scores_path = single_encoder_pruning

        if float(opt["alpha_vision"]) != 1.0:
            vision_encoder_pruning_info = os.path.join(
                'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
                f'sparsity_{opt["sparsity_vision"]}_alpha_{opt["alpha_vision"]}',
                vision_encoder_layers,
            )
            vision_encoder_scores_path = os.path.join(
                'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
                vision_encoder_layers
            )
        else:
            single_encoder_pruning = 'TEXT_ENCODER_ONLY'
            opt['vision_encoder_scorer'] = None
            single_encoder_pruning_scores_path = single_encoder_pruning

        if float(opt["alpha_text"]) != 1.0 and float(opt["alpha_vision"]) != 1.0:
            final_pruning_info = os.path.join(
                text_encoder_pruning_info,
                vision_encoder_pruning_info,
            )
            final_pruning_scores_path = os.path.join(
                text_encoder_scores_path,
                vision_encoder_scores_path
            )
        elif float(opt["alpha_text"]) == 1.0:
            final_pruning_info = os.path.join(
                single_encoder_pruning,
                vision_encoder_pruning_info,
            )
            final_pruning_scores_path = os.path.join(
                single_encoder_pruning_scores_path,
                vision_encoder_scores_path
            )
        else:
            final_pruning_info = os.path.join(
                single_encoder_pruning,
                text_encoder_pruning_info,
            )
            final_pruning_scores_path = os.path.join(
                single_encoder_pruning_scores_path,
                text_encoder_scores_path
            )

        opt['save_path'] = os.path.join(
            base_save_dir,
            final_pruning_info,
            concept_mode
        )
        opt['path_scores'] = os.path.join(
            SCORES_PATH,
            opt['model_name'],
            f'seed_{opt["seed"]}',
            str(opt['pruning_dataset_size']),
            final_pruning_scores_path,
            'saved_scores'
        )
        os.makedirs(opt['path_scores'], exist_ok=True)
        if opt['plot']:
            opt['plot_path'] = os.path.join(opt['save_path'], 'plots')
            os.makedirs(opt['plot_path'], exist_ok=True)

    opt['model_info'] = config.VLM_MODELS[opt['model_name']]
    opt['device'] = 'cuda:0'
    os.makedirs(opt['save_path'], exist_ok=True)
    utils.set_deterministic(opt['seed'])
    return opt 

def zero_shot():
    parser = argparse.ArgumentParser(description='CLIP retrieval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model_name', type=str, default='ViT-B/32', choices=list(config.VLM_MODELS.keys()), help='Model name')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--inference_dataset', type=str, choices=list(config.ZERO_SHOT_DATASETS.keys()), help='Inference dataset to use')
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'prune'], help='Mode to use')
    parser.add_argument('--v_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(vision_scorers)).keys()), help='Vision scorer to use')
    parser.add_argument('--t_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(text_scorers)).keys()), help='Scorer to use')
    parser.add_argument('--sparsity_vision', type=float, default=0.999, help='Sparsity for vision encoder')
    parser.add_argument('--sparsity_text', type=float, default=0.999, help='Sparsity for text encoder')
    parser.add_argument('--alpha_vision', type=float, default=0, help='Alpha for vision encoder')
    parser.add_argument('--alpha_text', type=float, default=0, help='Alpha for text encoder')
    parser.add_argument('--text_encoder_layers', nargs='+', default=None, help='Layers to prune')
    parser.add_argument('--vision_encoder_layers', nargs='+', default=None, help='Layers to prune')
    opt = vars(parser.parse_args())

    opt['inference_dataset'] = config.ZERO_SHOT_DATASETS[opt['inference_dataset']]
    base_save_dir = os.path.join('results', opt['inference_dataset']['name'], 'CLIP_zero_shot', opt['model_name'])
    if opt['mode'] == 'original':
        opt['save_path'] = os.path.join(base_save_dir, 'original')
    else:
        assert set(opt['vision_encoder_layers']).issubset(config.VLM_MODELS[opt['model_name']]['layers']), 'Invalid vision encoder layers, please select from: ' + ', '.join(config.VLM_MODELS[opt['model_name']]['layers'])
        assert set(opt['text_encoder_layers']).issubset(config.VLM_MODELS[opt['model_name']]['layers']), 'Invalid text encoder layers, please select from: ' + ', '.join(config.VLM_MODELS[opt['model_name']]['layers'])
        
        opt['vision_encoder_scorer'] = {
            'name': opt['v_scorer'],
            'class': getattr(vision_scorers, opt['v_scorer']),
        }
        opt['vision_encoder_pruning_dataset'] = config.VISION_PRUNING_DTS['ViSU']
        opt['text_encoder_pruning_dataset'] = config.TEXT_PRUNING_DTS['ViSU']
        opt['text_encoder_scorer'] = {
            'name': opt['t_scorer'],
            'class': getattr(text_scorers, opt['t_scorer']),
        }
        text_encoder_layers = '' if opt['text_encoder_layers'] is None else 'T_layers_' + '_'.join(opt['text_encoder_layers'])
        vision_encoder_layers = '' if opt['vision_encoder_layers'] is None else 'V_layers_' + '_'.join(opt['vision_encoder_layers'])
        opt['pruning_dataset_size'] = 2048

        opt['concepts'] = ['all']
        # ~400 datapoints per concept
        opt['pruning_dataset_size'] = opt['pruning_dataset_size']*4
        
        if 'gradient' in opt['vision_encoder_scorer']['name'].lower() or 'gradient' in opt['text_encoder_scorer']['name'].lower():
            opt['get_pruners_fn'] = utils.get_pruners_gradient
            concept_mode = 'gradient'
            opt['text_encoder_pruning_dataset'] = config.VISION_PRUNING_DTS['ViSU']
        else:
            opt['get_pruners_fn'] = utils.get_pruners
            concept_mode = 'UWM'

        # if alpha == 1 the original encoder is used
        # change the save path accordingly
        if float(opt["alpha_text"]) != 1.0:
            text_encoder_pruning_info = os.path.join(
                'T_ENCODER_' + opt['text_encoder_scorer']['name'],
                text_encoder_layers,
                f'sparsity_{opt["sparsity_text"]}_alpha_{opt["alpha_text"]}',
            )
            text_encoder_scores_path = os.path.join(
                'T_ENCODER_' + opt['text_encoder_scorer']['name'],
                text_encoder_layers
            )
        else:
            single_encoder_pruning = 'VISION_ENCODER_ONLY'
            opt['text_encoder_scorer'] = None
            single_encoder_pruning_scores_path = single_encoder_pruning

        if float(opt["alpha_vision"]) != 1.0:
            vision_encoder_pruning_info = os.path.join(
                'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
                f'sparsity_{opt["sparsity_vision"]}_alpha_{opt["alpha_vision"]}',
                vision_encoder_layers,
            )
            vision_encoder_scores_path = os.path.join(
                'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
                vision_encoder_layers
            )
        else:
            single_encoder_pruning = 'TEXT_ENCODER_ONLY'
            opt['vision_encoder_scorer'] = None
            single_encoder_pruning_scores_path = single_encoder_pruning

        if float(opt["alpha_text"]) != 1.0 and float(opt["alpha_vision"]) != 1.0:
            final_pruning_info = os.path.join(
                text_encoder_pruning_info,
                vision_encoder_pruning_info,
            )
            final_pruning_scores_path = os.path.join(
                text_encoder_scores_path,
                vision_encoder_scores_path
            )
        elif float(opt["alpha_text"]) == 1.0:
            final_pruning_info = os.path.join(
                single_encoder_pruning,
                vision_encoder_pruning_info,
            )
            final_pruning_scores_path = os.path.join(
                single_encoder_pruning_scores_path,
                vision_encoder_scores_path
            )
        else:
            final_pruning_info = os.path.join(
                single_encoder_pruning,
                text_encoder_pruning_info,
            )
            final_pruning_scores_path = os.path.join(
                single_encoder_pruning_scores_path,
                text_encoder_scores_path
            )

        opt['save_path'] = os.path.join(
            base_save_dir,
            final_pruning_info,
            concept_mode
        )
        opt['path_scores'] = os.path.join(
            SCORES_PATH,
            opt['model_name'],
            'seed_0', # Update if seed is changed
            '8192', # Update if pruning dataset size is changed
            final_pruning_scores_path,
            'saved_scores'
        )
        os.makedirs(opt['path_scores'], exist_ok=True)

    opt['device'] = 'cuda:0'
    opt['model_info'] = config.VLM_MODELS[opt['model_name']]
    os.makedirs(opt['save_path'], exist_ok=True)

    utils.set_deterministic(0)
    return opt   

def captioning():
    parser = argparse.ArgumentParser(description='CLIP retrieval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--captioning_model', type=str, default='Llava-1.5-13B', choices=list(config.CAPTIONERS.keys()), help='Model name')
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'prune', 'safeclip', 'prune_safeclip'], help='Mode to use')
    parser.add_argument('--v_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(vision_scorers)).keys()), help='Vision scoring function to use')
    parser.add_argument('--sparsity_vision', type=float, default=0.999, help='Sparsity for vision encoder')
    parser.add_argument('--alpha_vision', type=float, default=0, help='Alpha for vision encoder')
    parser.add_argument('--vision_encoder_layers', nargs='+', default=None, help='Layers to prune')
    opt = vars(parser.parse_args())

    opt['captioner'] = config.CAPTIONERS[opt['captioning_model']]
    opt['inference_dataset'] = config.DATASETS['ViSU']
    base_save_dir = os.path.join('results', opt['inference_dataset']['name'], 'Captioning', opt['captioner']['name'])
    if opt['mode'] == 'original':
        opt['save_path'] = os.path.join(base_save_dir, 'original')
    elif opt['mode'] == 'prune' or opt['mode'] == 'prune_safeclip':
        assert opt["alpha_vision"] != 1.0, 'Alpha for vision encoder cannot be 1.0'
        
        opt['text_encoder_scorer'] = None
        if opt['mode'] == 'prune':
            opt['model_name'] = 'Clip'
        else:
            opt['model_name'] = 'SafeCLIP'
            opt['model_info'] = {
                'model_name': 'aimagelab/safeclip_vit-l_14_336',  # specific safe clip for llava 1.5-13B
                'tokenizer_name': 'openai/clip-vit-large-patch14-336',
                'preprocessor': 'openai/clip-vit-large-patch14-336',
                'layers': ['text_projection', 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
            }
            base_save_dir = os.path.join(base_save_dir, 'safeclip', 'prune')
        opt['vision_encoder_scorer'] = {
            'name': opt['v_scorer'],
            'class': getattr(vision_scorers, opt['v_scorer']),
        }
        opt['vision_encoder_pruning_dataset'] = config.VISION_PRUNING_DTS['ViSU']

        vision_encoder_layers = '' if opt['vision_encoder_layers'] is None else 'V_layers_' + '_'.join(opt['vision_encoder_layers'])
        opt['pruning_dataset_size'] = 2048

        opt['concepts'] = ['all']
        # ~400 datapoints per concept
        opt['pruning_dataset_size'] = opt['pruning_dataset_size']*4
        
        if not 'uninformedpruning' in opt['vision_encoder_scorer']['name'].lower() and \
            (
                'informedpruning' in opt['vision_encoder_scorer']['name'].lower() or 'safeclip' in opt['vision_encoder_scorer']['name'].lower()
            ):
            opt['get_pruners_fn'] = utils.get_pruners_gradient
            concept_mode = 'gradient'
        else:
            opt['get_pruners_fn'] = utils.get_pruners
            concept_mode = 'UWM'

        vision_encoder_pruning_info = os.path.join(
            'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
            f'sparsity_{opt["sparsity_vision"]}_alpha_{opt["alpha_vision"]}',
            vision_encoder_layers,
        )

        opt['save_path'] = os.path.join(
            base_save_dir,
            vision_encoder_pruning_info,
            concept_mode
        )
        final_pruning_scores_path = os.path.join(
            'T_ENCODER_' + opt['vision_encoder_scorer']['name'],
            'T_layers_out_proj',
            'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
            vision_encoder_layers
        )
        opt['path_scores'] = os.path.join(
            SCORES_PATH,
            'ViT-L14',
            final_pruning_scores_path,
            'saved_scores'
        )
    elif opt['mode'] == 'safeclip':
        opt['model_info'] = {
            'model_name': 'aimagelab/safeclip_vit-l_14_336',  # specific safe clip for llava 1.5-13B
            'tokenizer_name': 'openai/clip-vit-large-patch14-336',
            'preprocessor': 'openai/clip-vit-large-patch14-336',
            'layers': ['text_projection', 'k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
        }
        opt['save_path'] = os.path.join(base_save_dir, 'safeclip')

    opt['device'] = 'cuda:0'
    os.makedirs(opt['save_path'], exist_ok=True)

    utils.set_deterministic(0)
    return opt   

def evaluate_captioning():
    parser = argparse.ArgumentParser(description='CLIP retrieval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--llm', type=str, default='llama3-8B', choices=list(config.LLMS.keys()), help='Model name')
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'prune', 'safeclip', 'prune_safeclip'], help='Mode to use')
    parser.add_argument('--v_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(vision_scorers)).keys()), help='Vision scoring function to use')
    parser.add_argument('--sparsity_vision', type=float, default=0.999, help='Sparsity for vision encoder')
    parser.add_argument('--alpha_vision', type=float, default=0, help='Alpha for vision encoder')
    parser.add_argument('--vision_encoder_layers', nargs='+', default=None, help='Layers to prune')
    opt = vars(parser.parse_args())

    opt['LLM'] = config.LLMS[opt['llm']]

    base_save_dir = os.path.join('results', 'ViSU', 'Captioning', 'Llava-1.5-13B')
    if opt['mode'] == 'original':
        opt['save_path'] = os.path.join(base_save_dir, 'original')
    elif opt['mode'] == 'prune' or opt['mode'] == 'prune_safeclip':
        assert opt["alpha_vision"] != 1.0, 'Alpha for vision encoder cannot be 1.0'
        
        if opt['mode'] == 'prune_safeclip':
            base_save_dir = os.path.join(base_save_dir, 'safeclip', 'prune')

        opt['vision_encoder_scorer'] = {
            'name': opt['v_scorer'],
            'class': getattr(vision_scorers, opt['v_scorer']),
        }

        vision_encoder_layers = '' if opt['vision_encoder_layers'] is None else 'V_layers_' + '_'.join(opt['vision_encoder_layers'])
        vision_encoder_pruning_info = os.path.join(
            'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
            f'sparsity_{opt["sparsity_vision"]}_alpha_{opt["alpha_vision"]}',
            vision_encoder_layers,
        )

        if not 'uninformedpruning' in opt['vision_encoder_scorer']['name'].lower() and \
            (
                'informedpruning' in opt['vision_encoder_scorer']['name'].lower() or 'safeclip' in opt['vision_encoder_scorer']['name'].lower()
            ):
            concept_mode = 'gradient'
        else:
            concept_mode = 'UWM'

        opt['save_path'] = os.path.join(
            base_save_dir,
            vision_encoder_pruning_info,
            concept_mode
        )
    elif opt['mode'] == 'safeclip':
        opt['save_path'] = os.path.join(base_save_dir, 'safeclip')

    opt['seed'] = 0
    utils.set_deterministic(opt['seed'])
    return opt

def evaluate_captioning_API():
    parser = argparse.ArgumentParser(description='CLIP retrieval', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, default='original', choices=['original', 'prune', 'safeclip', 'prune_safeclip'], help='Mode to use')
    parser.add_argument('--v_scorer', type=str, default=None, choices=list(dict(inspect.getmembers(vision_scorers)).keys()), help='Vision scoring function to use')
    parser.add_argument('--sparsity_vision', type=float, default=0.999, help='Sparsity for vision encoder')
    parser.add_argument('--alpha_vision', type=float, default=0, help='Alpha for vision encoder')
    parser.add_argument('--vision_encoder_layers', nargs='+', default=None, help='Layers to prune')
    opt = vars(parser.parse_args())

    base_save_dir = os.path.join('results', 'ViSU', 'Captioning', 'Llava-1.5-13B')
    if opt['mode'] == 'original' or opt['mode'] == 'safeclip':
        opt['save_path'] = os.path.join(base_save_dir, opt['mode'])
    else:
        assert opt["alpha_vision"] != 1.0, 'Alpha for vision encoder cannot be 1.0'
        
        if opt['mode'] == 'prune_safeclip':
            base_save_dir = os.path.join(base_save_dir, 'safeclip', 'prune')

        opt['vision_encoder_scorer'] = {
            'name': opt['v_scorer'],
            'class': getattr(vision_scorers, opt['v_scorer']),
        }
        
        vision_encoder_layers = '' if opt['vision_encoder_layers'] is None else 'V_layers_' + '_'.join(opt['vision_encoder_layers'])
        vision_encoder_pruning_info = os.path.join(
            'V_ENCODER_' + opt['vision_encoder_scorer']['name'],
            f'sparsity_{opt["sparsity_vision"]}_alpha_{opt["alpha_vision"]}',
            vision_encoder_layers,
        )

        if not 'uninformedpruning' in opt['vision_encoder_scorer']['name'].lower() and \
            (
                'informedpruning' in opt['vision_encoder_scorer']['name'].lower() or 'safeclip' in opt['vision_encoder_scorer']['name'].lower()
            ):
            concept_mode = 'gradient'
        else:
            concept_mode = 'UWM'

        opt['save_path'] = os.path.join(
            base_save_dir,
            vision_encoder_pruning_info,
            concept_mode
        )

    opt['seed'] = 0
    utils.set_deterministic(opt['seed'])
    return opt