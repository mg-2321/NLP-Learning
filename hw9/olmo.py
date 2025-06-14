import argparse
import random
import re

import torch

from typing import Any, Iterable, Protocol

from tqdm import tqdm

from data import SSTClassificationDataset

Example = dict[str, Any]


# Using a Protocol to define the function signature for template functions
# This allows us to use any function that matches this signature as a template function
# Because we want `sentiment` to be optional, this goes beyond the basic `Callable` syntax for annotating functions
class TemplateFunction(Protocol):
    def __call__(self, review: str, sentiment: int | None = None) -> str: ...


def template_basic(review: str, sentiment: int | None = None) -> str:
    """
    Basic prompt template for sentiment classification.
    Takes one review, possibly with a sentiment label, and formats it into a string.
    The label `sentiment` will be provided for in-context examples, but not for the final examnple to be predicted.
    """
    # NB: by using {sentiment or ''} in this f-string, when `sentiment` is not provided,
    # the empty string will be put there instead, so this same method can be used for filled
    # and un-filled examples.
    return f"Review: {review}\nRating: {sentiment or ''}"


def template_complex(review: str, sentiment: int | None = None) -> str:
    # TODO: implement here!  You can try anything you'd like for the single-example template here.
    # Examples include specifying what the possible ratings are (1, 2, 3, 4, 5), or changing the formatting, or the wording.
    # In principle, any of these things can make a difference in the performance of the model.
    rating_str = f"{sentiment}" if sentiment is not None else ""
    return (
        "Movie Review Sentiment Analysis\n"
        "Please rate the sentiment of the following review on a scale from 1 to 5,\n"
        "where 1 = very negative, 3 = neutral, 5 = very positive.\n\n"
        f"Review: \"{review}\"\n"
        f"Sentiment rating: {rating_str}"
    )



def prompt_basic(
    new_review: str,
    examples: Iterable[Example] = [],
    prompt_template: TemplateFunction = template_basic,
) -> str:
    """
    Basic prompt that just concatenates in-context examples and then applies the template to a new review.
    """
    in_context_examples = ""
    if examples:
        in_context_examples = "\n\n".join(
            prompt_template(example["review"], example["label"]) for example in examples
        )
        in_context_examples += "\n\n"
    return f"{in_context_examples}{prompt_template(new_review)}"


def prompt_complex(
    new_review: str,
    examples: Iterable[Example] = [],
    prompt_template: TemplateFunction = template_basic,
) -> str:
    """
    Complex prompt that first includes an instruction / task description to the LM,
    then concatenates in-context examples, and then applies the template to a new review.
    """
    # TODO: implement here! You can try anything you'd like for the instruction string, as well as anything else in the total prompt.
    """
    Builds a full instruction-based prompt:
    1. Instruction
    2. In-context examples with full template
    3. New review unfilled
    """
    instruction = (
        "You are a helpful assistant that reads movie reviews and assigns them\n"
        "an integer sentiment rating from 1 (very negative) to 5 (very positive).\n"
        "Only output the single integer on its own line.\n"
    )
    in_context = ""
    if examples:
        in_context = "\n\n".join(
            prompt_template(ex["review"], ex["label"]) for ex in examples
        )
        in_context += "\n\n"
    # for the new review, we pass sentiment=None so the template leaves the slot blank
    new_prompt = prompt_template(new_review, None)
    return f"{instruction}\n{in_context}{new_prompt}"


def get_k_examples(
    dataset: SSTClassificationDataset,
    k: int = 5,
) -> list[Example]:
    """
    Get a list of k _random_ examples from the dataset.
    """
    example_indices = random.sample(range(len(dataset)), k)
    return [dataset[i] for i in example_indices]


def extract_rating(generation: str) -> int:
    """
    Given a string (a model generation), extract a rating (1, 2, 3, 4, or 5) from it.
    If no rating is found, return -1.
    """
    # TODO: implement here! You can try anything you'd like for the rating extraction as well.
    # Here's one concrete option: use a regex to find the first digit between 1 and 5 in the generation and return that (or -1).
    match = re.search(r"\b([1-5])\b", generation)
    if match:
        return int(match.group(1))
    else:
        return -1


def accuracy(predictions: list[int], labels: list[int]) -> float:
    return sum(
        predictions[idx] == labels[idx] for idx in range(len(predictions))
    ) / len(predictions)


if __name__ == "__main__":

    # argparse train and dev files for sentiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_reviews",
        type=str,
        default="/mnt/dropbox/24-25/574/data/sst/train-reviews.txt",
    )
    parser.add_argument(
        "--train_labels",
        type=str,
        default="/mnt/dropbox/24-25/574/data/sst/train-labels.txt",
    )
    parser.add_argument(
        "--dev_reviews",
        type=str,
        default="/mnt/dropbox/24-25/574/data/sst/dev-reviews-256.txt",
    )
    parser.add_argument(
        "--dev_labels",
        type=str,
        default="/mnt/dropbox/24-25/574/data/sst/dev-labels-256.txt",
    )
    parser.add_argument(
        "--hf_home", type=str, default="/mnt/dropbox/24-25/574/.cache/huggingface"
    )
    parser.add_argument("--model_name", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--seed", type=int, default=574)
    parser.add_argument("--batch_size", type=int, default=32)
    # in-context learning hyper-parameters
    parser.add_argument("--num_in_context_examples", type=int, default=0)
    parser.add_argument(
        "--template_method", type=str, choices=["basic", "complex"], default="basic"
    )
    parser.add_argument(
        "--prompt_method",
        type=str,
        choices=["basic", "complex"],
        default="basic",
    )
    # generation hyper-parameters
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    # it's not good practice to do these imports in main, but this order of
    # operations is needed in order to have transformers load models from a
    # shared cache directory instead of downloading the model separately for every user
    import os

    os.environ["HF_HOME"] = args.hf_home
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    # set seed
    set_seed(args.seed)

    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    olmo = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16  # , load_in_8bit=True
    )

    # get datasets
    sst_train = SSTClassificationDataset(args.train_reviews, args.train_labels)
    sst_dev = SSTClassificationDataset(args.dev_reviews, args.dev_labels)

    # Select the template method based on the argument
    template_method = (
        template_basic if args.template_method == "basic" else template_complex
    )
    # Select the prompt method based on the argument
    prompt_method = prompt_basic if args.prompt_method == "basic" else prompt_complex

    dev_size = len(sst_dev)
    batch_size = args.batch_size
    dev_predictions = []
    for start in tqdm(range(0, dev_size, batch_size)):
        end = min(start + batch_size, dev_size)
        # get the batch of examples from dev set
        batch_examples = sst_dev[start:end]
        # choose in-context examples from train set
        in_context_examples = get_k_examples(sst_train, args.num_in_context_examples)
        # get prompts, one per example
        batch_prompts = [
            prompt_method(example["review"], in_context_examples, template_method)
            for example in batch_examples
        ]
        # tokenize the batch of prompts
        # note that padding goes on the left, since we will generate from these
        batch_tokens = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, padding_side="left"
        )
        # generate responses from our model
        batch_response = olmo.generate(
            **batch_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        # decode integers -> tokens for the generations
        batch_generations = tokenizer.batch_decode(
            # batch_response has the prompt tokens concatenated with the generated tokens
            # so we slice it to get only the newly generated tokens
            # batch_tokens.input_ids has shape [batch_size, max_prompt_len], so batch_tokens.input_ids.shape[1] is `max_prompt_len`
            # if we start from there in the output, that will be where the new tokens begin
            batch_response[:, batch_tokens.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        # convert generations to labels for our task
        batch_ratings = [extract_rating(gen) for gen in batch_generations]
        dev_predictions.extend(batch_ratings)

        if start == 0:
            # for the batch of examples, print: the prompt, the generation, the extracted rating (predicted label), and the true label
            print("===== Example Baatch =====\n")
            for example, prompt, generation, rating in zip(
                batch_examples, batch_prompts, batch_generations, batch_ratings
            ):
                print(
                    f"Prompt:\n{prompt}\n\nGeneration:\n{generation}\n\nExtracted Label: {rating}\nTrue Label: {example['label']}\n\n"
                )
            print("==========================\n")

    print(
        f"TOTAL DEV ACCURACY: {accuracy(dev_predictions, [example["label"] for example in sst_dev])}"
    )
