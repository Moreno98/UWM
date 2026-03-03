from models.llama3.generation import Llama as LLama3
import os 

class Llama_3:
    def __init__(
        self,
        rank,
        opt
    ) -> None:
        # set RANK environment variable
        self.generator = LLama3.build(
            ckpt_dir=opt['LLM']['weights_path'],
            tokenizer_path=opt['LLM']['tokenizer_path'],
            max_seq_len=opt['LLM']['max_seq_len'],
            max_batch_size=opt['LLM']['batch_size'],
            model_parallel_size=opt['LLM']['model_parallel_size'],
            seed=opt['seed'],
        )
        self.force_answer_prompt = opt['LLM']['force_answer_prompt']
        self.max_gen_len = opt['LLM']['max_gen_len']
        self.temperature = opt['LLM']['temperature']
        self.top_p = opt['LLM']['top_p']
        self.SYSTEM_PROMPT = opt['LLM']['SYSTEM_PROMPT']

    def generate(self, sentences):
        dialogs = []
        for sentence in sentences:
            sentence = sentence.replace("\n","")
            dialogs.append(
                self.SYSTEM_PROMPT + [
                    {
                        'role': 'user',
                        'content': sentence
                    }
                ]
            )
        return self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
            force_answer_prompt=self.force_answer_prompt
        )