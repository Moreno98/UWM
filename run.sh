# --------- Retrieval results ---------
# Original ViT-L14 CLIP retrieval results
python retrieval.py --model_name ViT-L14 --batch_size 512 --mode original --inference_dataset ViSU 

# Safe-CLIP retrieval results
python retrieval.py --model_name Safe-CLIP --batch_size 512 --mode original --inference_dataset ViSU 

# UWM results
python retrieval.py --model_name ViT-L14 --batch_size 512 --mode prune --inference_dataset ViSU \
    --t_scorer UWM --sparsity_text 0.98 --alpha_text -1 --text_encoder_layers out_proj --text_encoder_pruning_dataset ViSU \
    --v_scorer UWM --sparsity_vision 0.98 --alpha_vision -1  --vision_encoder_layers fc2 --vision_encoder_pruning_dataset ViSU 

# Gradient Unsafe pruning baseline
python retrieval.py --model_name ViT-L14 --batch_size 512 --mode prune --inference_dataset ViSU \
    --t_scorer GradientUnsafe --sparsity_text 0.965 --alpha_text 0 --text_encoder_layers out_proj --text_encoder_pruning_dataset ViSU \
    --v_scorer GradientUnsafe --sparsity_vision 0.965 --alpha_vision 0  --vision_encoder_layers fc2 --vision_encoder_pruning_dataset ViSU 

# Gradient Safe CLIP pruning baseline
python retrieval.py --model_name ViT-L14 --batch_size 512 --mode prune --inference_dataset ViSU \
    --t_scorer GradientSafeCLIP --sparsity_text 0.995 --alpha_text 0 --text_encoder_layers out_proj --text_encoder_pruning_dataset ViSU \
    --v_scorer GradientSafeCLIP --sparsity_vision 0.995 --alpha_vision 0  --vision_encoder_layers fc2 --vision_encoder_pruning_dataset ViSU 


# ----- Knowledge preservation -- Zero-shot results -----
# Original ViT-L14 CLIP results -- Replace model name with the desired version
# We iterate across all supported datasets
for dataset in flowers102 caltech101 ImageNetV2 ImageNetA ImageNetR ImageNetSketch cifar10 cifar100 EuroSAT StanfordCars food oxfordpets sun397 aircraft ucf dtd
do
    python zero_shot.py --model_name ViT-L14 --batch_size 512 --inference_dataset $dataset --mode original
done

# Run zero-shot with specific pruning method (UWM in this case) to evaluate knowledge preservation across datasets
# To run other methods, please update their corresponding hyperparameters as above.
for dataset in flowers102 caltech101 ImageNetV2 ImageNetA ImageNetR ImageNetSketch cifar10 cifar100 EuroSAT StanfordCars food oxfordpets sun397 aircraft ucf dtd
do
    python zero_shot.py --model_name ViT-L14 --batch_size 512 --inference_dataset $dataset --mode prune \
                        --t_scorer UWM --sparsity_text 0.98 --alpha_text -1 --text_encoder_layers out_proj \
                        --v_scorer UWM --sparsity_vision 0.98 --alpha_vision -1  --vision_encoder_layers fc2
done

# ----- Captioning -----
# Captioning requires transformers 4.31.0 (you might need to create a separate environment for it).
python captioning.py --captioning_model llava-1.5-13B --mode original

# For captioning, we increase the sparsity to 0.915 as we are pruning only the vision encoder. 
python captioning.py --captioning_model llava-1.5-13B --mode prune --v_scorer UWM --sparsity_vision 0.915 --alpha_vision -1 --vision_encoder_layers fc2

# Evaluate SafeCLIP applied to LLaVA 
python captioning.py --captioning_model llava-1.5-13B --mode safeclip

# Evaluate UWM on SafeCLIP applied to LLaVA
python captioning.py --captioning_model llava-1.5-13B --mode prune_safeclip --v_scorer UWM --sparsity_vision 0.915 --alpha_vision -1 --vision_encoder_layers fc2

# We can evaluate captioning with LLM as a judge and the perspective API for NSFW content detection.
# LLM Evaluation
torchrun --master_port <PORT> --nproc_per_node 1 evaluate_captioning_LLM.py --llm llama3-8B --mode prune --v_scorer UWM --sparsity_vision 0.915 --alpha_vision -1 --vision_encoder_layers fc2

# Perspective API
python evaluate_captioning_perspective_API.py --mode prune --v_scorer UWM --sparsity_vision 0.915 --alpha_vision -1 --vision_encoder_layers fc2

