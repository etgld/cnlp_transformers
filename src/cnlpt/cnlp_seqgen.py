import argparse
from time import time

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# NB: for the types of models one is likely to use here
# all the documentation refers to techniques requiring BitsAndBytes as well as accelerate
# both of which can be installed via:
# pip install bitsandbytes accelerate
# (accelerate iirc should installed automatically when executing `pip install "transformers[torch]"`)

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

    # Getting a deprecation warning when using
    # `load_in_4bit`, Hf says to switch to a quantization config such as
    # https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.BitsAndBytesConfig
    # adding device map since that's a typical part of the
    # instructions for other cases
    model = AutoModelForCausalLM.from_pretrained(
        final_path,
        load_in_4bit=True,
        use_auth_token=True,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=args.attn_implementation,
    )
    # NB: check the model you are using to see if this
    # parameter is relevant
    # https://huggingface.co/docs/transformers/main/en/chat_templating#what-are-generation-prompts
    # pipeline = pipeline(
    #     "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
    # )
    # while notionally the pipeline interface takes care of the apply_chat_template
    # I'm starting to think it's more idiomatic to make that explicit
    # since I'm not sure that method always existed in previous versions of transformers
    end = time()
    print(f"Loading model took {end-start} seconds")

    # Have a flag for interactive mode but for now keep it to
    # current cnlpt functionality
    # while True:
    #     # Get user input:
    #     prompt = input("Enter the prompt you would like to give the model:\n>")
    #     if len(prompt) == 0:
    #         continue

    #     start = time()
    #     sequences = pipeline(
    #         prompt,
    #         do_sample=True,
    #         top_k=10,
    #         num_return_sequences=1,
    #         eos_token_id=tokenizer.eos_token_id,
    #         max_length=1000,
    #     )
    #     end = time()

    #     for seq_ind, seq in enumerate(sequences):
    #         print(f"************* Response {seq_ind} *************")
    #         print(f"{seq['generated_text']}")
    #         print(f"************* End response {seq_ind} ************\n\n")

    #     print(f"Response generated in {end-start} s")


if __name__ == "__main__":
    main()
