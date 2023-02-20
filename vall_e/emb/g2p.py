import argparse
import random
import string
from functools import cache
from pathlib import Path

import torch
from g2p_en import G2p
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from ipatok import tokenise as ipa_tokenizer


@cache
def _get_model():
    return G2p()


@cache
def _get_graphs(path):
    with open(path, "r") as f:
        graphs = f.read()
    return graphs


def encode(graphs: str) -> list[str]:
    g2p = _get_model()
    phones = g2p(graphs)
    ignored = {" ", *string.punctuation}
    return ["_" if p in ignored else p for p in phones]

def encode_multilang(words, lang_id, model, tokenizer, device='cuda:0', from_file=False) -> list[str]:
    if lang_id=='pt':
        lang_id = '<por-bz>'
    elif lang_id=='es':
        lang_id = '<spa>'
    #with open(path, "r") as f:
    #    words = f.read()
    #assert len(words) <= 150
    words = words.split()
    words = [f'{lang_id}: '+i for i in words]
    out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt')
    out = out.to(device)
    preds = model.generate(**out,num_beams=1)
    phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)[0]

    assembled = ''
    for w in phones.split():
        chars = " ".join(ipa_tokenizer(w))
        assembled+=chars+' _ '
    phones = assembled[:-3]
    return phones

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--suffix", type=str, default=".normalized.txt")
    parser.add_argument('--lang_id', type=str, required=True)
    args = parser.parse_args()

    paths = list(args.folder.rglob(f"*{args.suffix}"))
    random.shuffle(paths)
    device = 'cuda:0'

    if args.lang_id!='en':
        model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
        model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    for path in tqdm(paths):
        phone_path = path.with_name(path.stem.split(".")[0] + ".phn.txt")
        if phone_path.exists():
            continue
        if args.lang_id=='en':
            graphs = _get_graphs(path)
            phones = encode(graphs)
        else:
            try:
                phones = encode_multilang(path, args.lang_id, model, tokenizer)
            except AssertionError:
                print(f"Skipping file: {path}")
                continue
        with open(phone_path, "w") as f:
            f.write(" ".join(phones))
            del phones


if __name__ == "__main__":
    main()
