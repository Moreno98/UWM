import torch

# llava model imports
from models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from models.llava.conversation import conv_templates, SeparatorStyle
from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class Llava():
    def __init__(
        self,
        device,
        ckpt,
        model_base = None
    ):
        # Model
        disable_torch_init()
        self.model_name = get_model_name_from_path(ckpt)
        device_map = 'cpu' if device == 'cpu' else 'auto'
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(ckpt, model_base, self.model_name, device_map=device_map)
    
    def process_image(self, image):
        return self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    def vqa_batch(self, images, prompts):
        input_ids = []
        for prompt in prompts:
            ids, stopping_criteria, stop_str = self.get_prompt(f'Question: {prompt} Answer:')
            input_ids.append(ids)
        input_ids = torch.cat(input_ids, dim=0)

        output_ids = self.model.generate(
            input_ids,
            images=images,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()


    def get_prompt(self, qs):
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        return input_ids, stopping_criteria, stop_str

    @torch.no_grad()
    def vqa(self, **kwargs):
        if 'choices' in kwargs:
            qs = f'Question: {kwargs["question"]} Choices: {", ".join(kwargs["choices"])}. Answer:'
        else:
            qs = f'Question: {kwargs["question"]} Answer:'
            
        input_ids, stopping_criteria, stop_str = self.get_prompt(qs)
        image_tensor = kwargs['image']

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        
        return outputs