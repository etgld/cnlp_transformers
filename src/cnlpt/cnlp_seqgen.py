import argparse
import datetime
import logging
import os
import pathlib
import re
from collections import deque
from itertools import chain
from time import time
from typing import Callable, Deque, Dict, Iterable, List, Tuple, cast

import pandas as pd
import pytz
from tqdm import tqdm
from transformers import BitsAndBytesConfig, pipeline

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
    help="Check the `get_examples` method for the possible formats for now",
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
parser.add_argument("--fancy_output", action="store_true")
parser.add_argument("--model_name", choices=["llama2", "llama3", "mixtral", "qwen2"])


parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="1 for classification, on the order of 128 for BIO, on the order of 1024 for free text analysis and explanation",
)
parser.add_argument(
    "--query_files",
    nargs="+",
    default=[],
    help="TSVs for now, JSON or whatever else eventually",
)
parser.add_argument(
    "--query_dir",
    help="TSVs for now, JSON or whatever else eventually",
)
parser.add_argument("--output_dir", type=str)

name2path = {
    "llama2": "/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
    "llama3": "/lab-share/CHIP-Savova-e2/Public/resources/Meta-Llama-3-8B-Instruct/",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2": "Qwen/Qwen2-1.5B-Instruct",
}

# {role: {system|user|assistant}, content: ...}
Message = Dict[str, str]

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def main() -> None:
    args = parser.parse_args()
    final_path = ""
    if args.model_name is not None:
        final_path = name2path[args.model_name]
    else:
        final_path = args.model_path
    logger.info(f"Loading tokenizer and model for model name {final_path}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit
    )

    # Can't really find an idiomatic way to do this with
    # Hf Datasets.  My guess is part of this has to
    # do with best practices (or lack thereof?)
    # around how to best pad/trunctate input length
    system_prompt = get_system_prompt(args.prompt_file)
    if len(args.query_files) > 0:
        fn_to_queries = {q_fn: get_queries(q_fn) for q_fn in args.query_files}
    elif args.query_dir is not None and len(os.listdir(args.query_dir)) > 0:
        fn_to_queries = {q_fn: get_queries(q_fn) for q_fn in get_files(args.query_dir)}
    else:
        ValueError("Need at least one file to make queries")
        fn_to_queries = {}

    def few_shot_with_examples(
        examples: Iterable[Tuple[str, str]]
    ) -> Callable[[str, str], List[Message]]:
        def _few_shot_prompt(s, q):
            return few_shot_prompt(system_prompt=s, query=q, examples=examples)

        return _few_shot_prompt

    if args.examples_file is not None:
        examples = get_examples(args.examples_file)
        if len(examples) > 0:
            get_prompt = few_shot_with_examples(examples=examples)

        else:
            ValueError("Empty examples file")

            get_prompt = empty_prompt
    elif args.sample_document is not None and args.sample_answer is not None:
        example = get_document_level_example(args.sample_document, args.sample_answer)
        if all(len(ex) > 0 for ex in example):
            get_prompt = few_shot_with_examples(examples=(example,))
        else:
            ValueError("Empty sample document and/or empty sample answer")

            get_prompt = empty_prompt
    else:
        get_prompt = zero_shot_prompt
    start = time()
    # auth tokens for things like Mixtral
    seqgen_pipe = pipeline(
        "text-generation",
        model=final_path,
        # device="cuda",
        # use_auth_token=True,
        model_kwargs={"quantization_config": quantization_config, "device_map": "auto"},
    )
    end = time()
    logger.info(f"Loading model took {end-start} seconds")
    for q_fn, queries in fn_to_queries.items():
        out_dir = f"{args.output_dir}/{basename_no_ext(q_fn)}/{final_path.split('/')[-1]}/{basename_no_ext(args.prompt_file)}/{basename_no_ext(args.examples_file)}"
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        current_time = datetime.datetime.now(pytz.timezone("America/New_York"))
        out_fn = f"{current_time.strftime('%y-%m-%d_%H:%M')}.txt"
        out_path = os.path.join(out_dir, out_fn)
        for index, query in enumerate(tqdm(queries)):
            prompt_messages = get_prompt(system_prompt, query)
            outputs = seqgen_pipe(
                prompt_messages,
                max_new_tokens=args.max_new_tokens,
            )
            # to explain the arbitrary access ( huggingface doesn't do this naturally )
            # [0] since we're feeding the queries sequentially and as a result there's
            # only one element in the output list,
            # ["generated_text"] since everything is under that
            # [-1] since everything before that is the prompt structure
            # ["content"] since that's where the actual answer is
            answer = outputs[0]["generated_text"][-1]["content"]
            with open(out_path, mode="at", encoding="utf-8") as out_f:
                if args.fancy_output:
                    out_f.write(structure_response(index, query, answer))
                else:
                    out_f.write(answer + "\n")
    logger.info("Finished writing results")


def empty_prompt(system_prompt: str, query: str) -> List[Message]:
    return []


def structure_response(index: int, query: str, answer: str) -> str:
    return f"Query {index}:\n{query}\nAnswer:\n{answer}\n\n"


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_system_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, mode="rt", encoding="utf-8") as f:
        raw_prompt = f.read()
        # cleaned_prompt = " ".join(raw_prompt.strip().split())
        # return cleaned_prompt
        return raw_prompt


def get_queries(queries_file_path: str) -> Iterable[str]:
    # NB, this will retrieve the extension with the "." at the front
    # e.g. ".txt" rather than "txt"
    suffix = pathlib.Path(queries_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pd.read_csv(queries_file_path, sep="\t")
            queries = cast(Iterable[str], full_dataframe["query"])
        case ".txt" | "":
            with open(queries_file_path, mode="rt") as qf:
                query = qf.read()
                queries = (query,)
        case _:
            ValueError(f"Presently unsupported query format {suffix}")
            queries = []
    return queries


def get_examples(examples_file_path: str) -> List[Tuple[str, str]]:
    suffix = pathlib.Path(examples_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pd.read_csv(examples_file_path, sep="\t")
            queries = cast(Iterable[str], full_dataframe["query"])
            responses = cast(Iterable[str], full_dataframe["response"])
            examples = list(zip(queries, responses))
        case ".txt" | "":
            examples = parse_input_output(examples_file_path)
        case _:
            ValueError(f"Presently unsupported examples file format {suffix}")
            examples = []
    return examples


def parse_input_output(examples_file_path: str) -> List[Tuple[str, str]]:
    def parse_example(raw_example: str) -> Tuple[str, str]:
        result = tuple(
            elem.strip()
            for elem in re.split("input:|output:", raw_example)
            if len(elem.strip()) > 0
        )
        assert len(result) == 2
        return result

    with open(examples_file_path, mode="rt", encoding="utf-8") as ef:
        raw_str = ef.read()
        return [
            parse_example(example.strip())
            for example in raw_str.split("\n\n")
            if len(example.split()) > 0
        ]


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


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


if __name__ == "__main__":
    main()
