import argparse
import pathlib
from collections import deque
from time import time
from typing import Callable, Deque, Dict, Iterable, List, Tuple, cast
from itertools import chain
from tqdm import tqdm
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
)
parser.add_argument(
    "--sample_document",
    type=str,
)
parser.add_argument(
    "--sample_answer",
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
parser.add_argument(
    "--queries_file", type=str, help="TSVs for now, JSON or whatever else eventually"
)
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
    print(f"Loading tokenizer and model for model name {final_path}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit
    )

    model_answers: Deque[Tuple[str,]] = deque()
    # Can't really find an idiomatic way to do this with
    # Hf Datasets.  My guess is part of this has to
    # do with best practices (or lack thereof?)
    # around how to best pad/trunctate input length
    system_prompt = get_system_prompt(args.prompt_file)
    queries = get_queries(args.queries_file)

    def few_shot_with_examples(
        examples: Iterable[Tuple[str, str]]
    ) -> Callable[[str, str], List[Message]]:
        def _few_shot_prompt(s, q):
            return few_shot_prompt(system_prompt=s, query=q, examples=examples)

        return _few_shot_prompt

    def empty_prompt(system_prompt: str, query: str) -> List[Message]:
        return []

    if args.examples_file is not None:
        examples = get_examples(args.examples_file)
        if len(examples) > 0:
            get_prompt = few_shot_with_examples(examples=examples)

        else:
            ValueError("Empty examples file")

            get_prompt = empty_prompt
    elif args.sample_document is not None and args.sample_answer is not None:
        example = get_document_level_example(args.sample_document, args.sample_answer)
        if all(example):
            get_prompt = few_shot_with_examples(examples=(example,))
        else:
            ValueError("Empty sample document and/or empty sample answer")

            get_prompt = empty_prompt
    else:
        get_prompt = zero_shot_prompt
    start = time()
    # auth tokens for things like Mixtral
    tokenizer = AutoTokenizer.from_pretrained(final_path, use_auth_token=True)

    model = AutoModelForCausalLM.from_pretrained(
        final_path,
        use_auth_token=True,
        device_map="auto",
        quantization_config=quantization_config,
        # apparently this isn't idiomatic and you're
        # supposed to load this via the model config
        # attn_implementation=args.attn_implementation,
    )
    end = time()
    print(f"Loading model took {end-start} seconds")
    for query in tqdm(queries):
        prompt_messages = get_prompt(system_prompt, query)
        input_ids = tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, return_tensors="pt"
        ).cuda()
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        gen_text = tokenizer.batch_decode(
            outputs.detach().cpu().numpy()[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        model_answers.append((" ".join(gen_text.strip().split()),))
    output_df = pd.DataFrame.from_records(model_answers, columns=["answers"])
    output_df.to_csv(args.output_file, index=False, sep="\t")
    print("Finished writing results")


def get_system_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, mode="rt", encoding="utf-8") as f:
        raw_prompt = f.read()
        cleaned_prompt = " ".join(raw_prompt.strip().split())
        return cleaned_prompt


def get_queries(queries_file_path: str) -> Iterable[str]:
    # NB, this will retrieve the extension with the "." at the front
    # e.g. ".txt" rather than "txt"
    suffix = pathlib.Path(queries_file_path).suffix.lower()
    match suffix:
        case ".tsv":
            full_dataframe = pd.read_csv(queries_file_path, sep="\t")
            queries = cast(Iterable[str], full_dataframe["query"])
        case ".txt":
            with open(queries_file_path, mode="rt") as qf:
                query = qf.read()
                queries = (query,)
        case _:
            ValueError(f"Presently unsupported query format {suffix}")
            queries = ("",)
    return queries


def get_examples(examples_file_path: str) -> List[Tuple[str, str]]:
    full_dataframe = pd.read_csv(examples_file_path, sep="\t")
    queries = cast(Iterable[str], full_dataframe["query"])
    responses = cast(Iterable[str], full_dataframe["response"])
    return list(zip(queries, responses))


def get_document_level_example(
    sample_document_path: str, sample_answer_path: str
) -> Tuple[str, str]:
    with open(sample_document_path, mode="rt", encoding="utf-8") as sample_document:
        # not normalizing newlines since those might be useful
        query = sample_document.read()
    sample_answer_dataframe = pd.read_csv(sample_answer_path, sep="\t")
    # specific to earlier use-case etc but for now
    answer = "\n".join(cast(Iterable[str], sample_answer_dataframe["query"]))
    return (query, answer)


def zero_shot_prompt(system_prompt: str, query: str) -> List[Message]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return messages


def few_shot_prompt(
    system_prompt: str, query: str, examples: Iterable[Tuple[str, str]]
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
