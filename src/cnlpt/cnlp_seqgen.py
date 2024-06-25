import argparse
from time import time
from typing import Dict, List, Tuple
from itertools import chain
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
)
parser.add_argument("--prompt_file", type=str)
parser.add_argument(
    "--model_path",
    type=str,
    default="/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
)

parser.add_argument(
    "--attn_implementation",
    type=str,
    default="spda",
    choices=["spda", "flash_attention_2"],
)

parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--model_name", choices=["llama2", "llama3", "mixtral", "qwen2"])


parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="1 for classification, on the order of 128 for BIO, on the order of 1024 for free text analysis and explanation",
)
parser.add_argument("--input_file", type=str, help="TSVs for now")
parser.add_argument("--output_file", type=str)

name2path = {
    "llama2": "/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
    "llama3": "/lab-share/CHIP-Savova-e2/Public/resources/Meta-Llama-3-8B-Instruct/",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2": "Qwen/Qwen2-1.5B-Instruct",
}

# {role: {system|user|assistant}, content: ...}
Message = Dict[str, str]


def main() -> None:
    args = parser.parse_args()
    final_path = ""
    if args.model_name is not None:
        final_path = name2path[args.model_name]
    else:
        final_path = args.model_path
    print(f"Loading tokenizer and model for model name {args.model_name}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit
    )
    start = time()
    # auth tokens for things like Mixtral
    tokenizer = AutoTokenizer.from_pretrained(final_path, use_auth_token=True)

    model = AutoModelForCausalLM.from_pretrained(
        final_path,
        load_in_4bit=True,
        use_auth_token=True,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=args.attn_implementation,
    )
    end = time()
    print(f"Loading model took {end-start} seconds")
    few_shot_llama_answers = []
    # Can't really find an idiomatic way to do this with
    # Hf Datasets.  My guess is part of this has to
    # do with best practices (or lack thereof?)
    # around how to best pad/trunctate input length
    for query in tqdm(queries):
        few_shot_prompt_messages = few_shot_prompt(
            PROMPT, query, few_shot_prompts
        )
        input_ids = tokenizer.apply_chat_template(
            few_shot_prompt_messages, tokenize=True, return_tensors="pt"
        ).cuda()
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        gen_text = tokenizer.batch_decode(
            outputs.detach().cpu().numpy()[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        few_shot_llama_answers.append(gen_text.strip())


def zero_shot_prompt(system_prompt: str, query: str) -> List[Message]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return messages


def few_shot_prompt(
    system_prompt: str, query: str, examples: List[Tuple[str, str]]
) -> List[Message]:
    def message_pair(ex_query: str, ex_answer: str) -> Tuple[Message, ...]:
        return {"role": "user", "content": ex_query}, {
            "role": "assistant",
            "content": ex_answer,
        }

    few_shot_examples = chain.from_iterable(
        message_pair(ex_query=ex_query, ex_answer=ex_answer)
        for ex_query, ex_answer in examples
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *few_shot_examples,
        {"role": "user", "content": query},
    ]
    return messages


if __name__ == "__main__":
    main()
