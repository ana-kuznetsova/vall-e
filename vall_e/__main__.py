import argparse
from pathlib import Path

import torch
from einops import rearrange

from .emb import g2p, qnt
from utils import to_device
from .emb.g2p import encode_multilang
from transformers import T5ForConditionalGeneration, AutoTokenizer
from .vall_e.nar import NAR
from .config import cfg
from .vall_e import get_model


def main():
    parser = argparse.ArgumentParser("VALL-E TTS")
    parser.add_argument("text")
    parser.add_argument("reference", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--ar-ckpt", type=Path, default="zoo/ar.pt")
    parser.add_argument("--nar-ckpt", type=Path, default="zoo/nar.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    
    #ar = torch.load(args.ar_ckpt).to(args.device)
    #nar = NAR(num_qnts)
    #nar = get_model(cfg.model)
    #state_dict = torch.load(args.nar_ckpt)['module']
    #print(torch.load(args.nar_ckpt)['data_sampler'])

    #nar.load_state_dict(state_dict)
    #nar = nar.to(args.device)
    #nar.eval()
    #symmap = {'</s>': 1, '<s>': 2, '_': 3, 'a': 4, 'aː': 5, 'b': 6, 'd': 7, 'd͡': 8, 'd͡z': 9, 'd͡ʒ': 10, 'e': 11, 'ẽ': 12, 'f': 13, 'h': 14, 'i': 15, 'ĩ': 16, 'j': 17, 'j̃': 18, 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'o': 23, 'õ': 24, 'p': 25, 's': 26, 't': 27, 't͡': 28, 't͡ʃ': 29, 'u': 30, 'ũ': 31, 'v': 32, 'w': 33, 'w̃': 34, 'z': 35, 'ɐ': 36, 'ɐ̃': 37, 'ɔ': 38, 'ɔ̃': 39, 'ɛ': 40, 'ɛ̃': 41, 'ɡ': 42, 'ɲ': 43, 'ɹ': 44, 'ɾ': 45, 'ʁ': 46, 'ʃ': 47, 'ʎ': 48, 'ʐ': 49, 'ʒ': 50, 'ʼ': 51}

    #symmap = nar.phone_symmap
    #symmap = ar.phone_symmap

    #proms = qnt.encode_from_file(args.reference)
    #proms = rearrange(proms, "1 l t -> t l")

    #encoded_phns = encode_multilang(text.split(), 'pt', model, tokenizer)

    ar = torch.load(args.ar_ckpt).to(args.device)
    nar = torch.load(args.nar_ckpt).to(args.device)

    symmap = ar.phone_symmap

    proms = qnt.encode_from_file(args.reference)
    proms = rearrange(proms, "1 l t -> t l")

    #phns = torch.tensor([symmap[p] for p in g2p.encode(args.text)])
    with open('g2p_inf/eg1.txt', 'r') as fo:
        encoded_phns = fo.read()

    phns = torch.tensor([symmap[p] for p in encoded_phns.split()])

    proms = to_device(proms, args.device)
    phns = to_device(phns, args.device)

    resp_list = ar(text_list=[phns], proms_list=[proms])
    resps_list = [r.unsqueeze(-1) for r in resp_list]

    resps_list = nar(text_list=[phns], proms_list=[proms], resps_list=resps_list)
    qnt.decode_to_file(resps=resps_list[0], path=args.out_path)
    print(args.out_path, "saved.")

    

if __name__ == "__main__":
    main()
