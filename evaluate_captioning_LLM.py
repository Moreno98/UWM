import utils.LLMs as LLMs
import utils.arg_parse as arg_parse
import os
from tqdm import tqdm

def main(opt):
    llm = LLMs.Llama_3(
        rank = None,
        opt = opt,
    )

    settings = ['captions_unsafe_input.txt']

    for setting in tqdm(settings, position=0, leave=True, desc='Settings'):
        # open captioning file
        captions = []
        with open(os.path.join(opt['save_path'], setting), 'r') as f:
            lines = f.readlines()
            for line in lines:
                caption = line.split(';', 1)[1].strip()
                captions.append(caption)
        
        size = len(lines)
        results = {
            'safe': 0,
            'unsafe': 0,
        }
        wrong_answers = []
        
        # batch_size 
        batch_size = opt['LLM']['batch_size']
        for i in tqdm(range(0, size, batch_size), position=1, leave=False, desc='Running LLM'):
            llm_output = llm.generate(captions[i:i+batch_size])
            for output in llm_output:
                output = output['generation']['content'].lower().strip()
                if output in results.keys():
                    results[output] += 1
                else:
                    wrong_answers.append((caption, output['generation']['content']))

        with open(os.path.join(opt['save_path'], setting.split('.')[0]+"_results.txt"), 'w') as f:
            f.write(f"Safe: {results['safe']}/{size}\n")
            f.write(f"Unsafe: {results['unsafe']}/{size}\n")
            f.write(f"Safe percentage: {results['safe']/size}\n")
            f.write(f"Unsafe percentage: {results['unsafe']/size}\n")
        
        with open(os.path.join(opt['save_path'], setting.split('.')[0]+"_wrong_answers.txt"), 'w') as f:
            for caption, llm_output in wrong_answers:
                f.write(f"Caption: {caption}\tLLM output: {llm_output}\n")

if __name__ == '__main__':
    opt = arg_parse.evaluate_captioning()
    main(opt)