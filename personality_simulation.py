from dotenv import load_dotenv
import numpy as np
from functools import reduce
import os
import json
import tqdm
import torch
import random
import argparse
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BIG5
from utils import save_to_json, find_answer_from_string
from repeng_plus import ControlVector, ControlModel, DatasetEntry
from personality_eval import score_report

user_tag, asst_tag = "[INST]", "[/INST]"

load_dotenv()
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
login(token=HUGGING_FACE_TOKEN)


def make_dataset(template: str, positive_personas: list[str], negative_personas: list[str], suffix_list: list[str]) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(positive_personas, negative_personas):
            positive_template = template.format(persona=positive_persona)
            negative_template = template.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{user_tag} {positive_template} {asst_tag} {suffix}",
                    negative=f"{user_tag} {negative_template} {asst_tag} {suffix}",
                )
            )
    return dataset


def make_ctrl_vector(model, tokenizer, dataset):
    model.reset()
    ctrl_vector = ControlVector.train(model, tokenizer, dataset)
    return ctrl_vector


def make_personality_vectors(make_dataset_template_fn):
    # load the templates for making datasets.
    with open(make_dataset_template_fn) as f:
        output_suffixes = json.load(f)
    truncated_output_suffixes = [tokenizer.convert_tokens_to_string(tokens[:i]) for tokens in (
        tokenizer.tokenize(s) for s in output_suffixes) for i in range(1, len(tokens))]

    # create personality vectors for each BIG-5 dimension.
    ctrl_vectors = {dim: {} for dim in BIG5}
    for personality_dim, all_adj_pairs in BIG5.items():
        # create a vector for each adj pair.
        print(f'training {personality_dim} vectors...')
        for neg_adj, pos_adj in all_adj_pairs:
            dataset = make_dataset(
                "Act as if you're extremely {persona}.",
                [pos_adj], [neg_adj], truncated_output_suffixes)
            ctrl_vec = make_ctrl_vector(model, tokenizer, dataset)
            ctrl_vectors[personality_dim][pos_adj] = ctrl_vec

    return ctrl_vectors


def self_introduction(model, vector, scalar, settings, return_perplexity=True):
    model.reset()
    model.set_control(vector, scalar)

    # prompt.
    prompt = f"{user_tag} Pretend that you're a real human and talk about your favorite activity in your free time. {asst_tag}"

    input_ids = tokenizer(
        prompt, return_tensors="pt").to(model.device)

    # get response.
    outputs = model.generate(**input_ids, **settings,  # max_new_tokens=5,
                             return_dict_in_generate=True, output_scores=True)
    res = tokenizer.decode(outputs.sequences.squeeze()).strip()
    res = res.split(asst_tag)[1]

    # get perplexity
    if return_perplexity:
        transition_scores = model.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = input_ids['input_ids'].size()[1]
        generated_tokens = outputs.sequences[:, input_length:]
        assert generated_tokens.size()[1] == transition_scores.size()[1]
        probs = np.exp(transition_scores[0].cpu().numpy())
        ppx = calculate_perplexity(probs)
    else:
        ppx = None

    model.reset()
    return res, ppx


def calculate_perplexity(probabilities):
    log_probabilities = np.log2(probabilities)
    average_log_probability = np.mean(log_probabilities)
    perplexity = 2 ** (-average_log_probability)
    return perplexity


def load_questions(questionnaire_fn):
    def _convert_statement(text):
        if not text.startswith('I '):
            text = f'I {text[0].lower()}{text[1:]}'
        if not text.endswith('.'):
            text = text + '.'
        return text

    with open(questionnaire_fn, 'r') as F:
        data = json.load(F)

    for d in data:
        d['statement'] = _convert_statement(d['statement'])

    return data


def personality_test(questions, model, vector, scalar, settings, target_dim=None):
    model.reset()
    model.set_control(vector, scalar)

    answers = {}
    for q_num, question in tqdm.tqdm(enumerate(questions)):
        # skip the question.
        if target_dim is not None and target_dim != question['dimension']:
            continue

        # prompt.
        statement = question['statement']
        #persona = "For the following task, respond in a way that matches this description: “My favorite food is mushroom ravioli. I've never met my father. My mother works at a bank. I work in an animal shelter."
        prompt = f"{user_tag} Pretend that you are a real human and evaluate the statement, {statement}. "
        prompt += 'Please rate how accurately this describes you on a scale from 1 to 5. (where 1="very inaccurate", 2="moderately inaccurate", 3="neither accurate nor inaccurate", 4="moderately accurate", and 5="very accurate").'
        prompt += f'Please answer using EXACTLY one of the following: 1, 2, 3, 4, or 5. {asst_tag}'
        prompt += ' I would rate this statement as '

        # get response.
        input_ids = tokenizer(
            prompt, return_tensors="pt").to(model.device)

        choice = None
        for _ in range(2):
            res = tokenizer.decode(model.generate(
                **input_ids, **settings).squeeze())

            res = res.split(asst_tag)[1]
            choice = find_answer_from_string(res)

        answers[q_num] = {'statement': question['statement'], "dimension": question['dimension'],
                          "math": question['math'], "answer": res, "choice": choice}

    model.reset()
    return answers


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument(
        '--make_dataset_template', default='./data/all_truncated_outputs.json')
    parser.add_argument(
        '--questionnaire_fn', default='./data/personality_tests/ipip_bffm.json')
    parser.add_argument('--n_adj', type=int, default=12)
    parser.add_argument('--n_trial', type=int, default=100)
    parser.add_argument('-p', '--output_dir', default='./tmp')
    args = parser.parse_args()

    # load tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float16)

    model = model.to("cuda" if torch.cuda.is_available(
    ) else "mps:0" if torch.backends.mps.is_available() else "cpu")
    model = ControlModel(model, list(range(-5, -18, -1)))

    # get personality vectors for each BIG-5 dimension.
    ctrl_vectors = make_personality_vectors(args.make_dataset_template)

    # settings.
    VEC_RANGE = (-2.5, 2.5)
    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "temperature": 1.0, "top_p": 1.0,
        "repetition_penalty": 1.2,
        "max_new_tokens": 256,
    }

    # load questionnaire.
    questions = load_questions(args.questionnaire_fn)

    # experiments.
    for personality_dim, all_adj_pairs in BIG5.items():
        results = []
        for exp_i in range(args.n_trial):
            # randomly sample vector strength from range.
            scalar = random.uniform(*VEC_RANGE)

            # make control vector.
            # randomly select N_ADJ adjective pairs.
            n_adj = min(args.n_adj, len(all_adj_pairs))
            adj_pairs = random.sample(
                all_adj_pairs, n_adj)
            positive_list = [pair[1] for pair in adj_pairs]
            negative_list = [pair[0] for pair in adj_pairs]
            print(positive_list, negative_list)
            ctrl_vec = reduce(lambda x, y: x + y,
                              [ctrl_vectors[personality_dim][adj] for adj in positive_list])
            # scale the vector properly.
            scalar_scaled = scalar / n_adj

            print(
                f'======== {personality_dim}x{round(scalar, 2)} ========')

            # self introduction.
            self_intro, ppx = self_introduction(
                model, ctrl_vec, scalar_scaled, settings)
            print(f'[{round(ppx, 3)}] {self_intro}')

            # do ipip test.
            answers = personality_test(
                questions, model, ctrl_vec, scalar_scaled, settings, target_dim=personality_dim)
            #scores = score_report(answers)
            # print(scores)

            # save to results.
            results.append({'agent': (personality_dim, scalar),
                            'self_intro': self_intro, 'perplexity': ppx,
                            'answers': answers})

        # save to file.
        output_fn = os.path.join(
            args.output_dir, f'{personality_dim}.json')
        save_to_json(results, output_fn)
