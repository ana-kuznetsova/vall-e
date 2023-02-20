import pandas as pd
import argparse
import shutil
from cvutils import Alphabet
from tqdm.contrib import tzip
from transformers import T5ForConditionalGeneration, AutoTokenizer
from ipatok import tokenise as ipa_tokenizer

'''
1. Preprocess csv
2. Normalize text
3. Create data directory
'''

def normalize_text(text, lang_id, alphabet):
    text_clean = ''.join([ch for ch in text.lower() if ch in alphabet])
    return text_clean

def encode_multilang(words, lang_id, model, tokenizer, device='cuda:0', from_file=False):
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
    phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)

    assembled = ''
    for w in phones:
        chars = " ".join(ipa_tokenizer(w))
        assembled+=chars+' _ '
    phones = assembled[:-3]
    return phones

def create_spk_map(df):
    client_ids = set(df['client_id'].values)
    spk_map = {str(s):i for i, s in enumerate(client_ids)}
    return spk_map

def cv_from_validated(args):
    df = pd.read_csv(f"{args.inp_dir}/{args.lang_id}/validated.tsv", sep='\t')
    out_dir = f"{args.out_dir}/{args.lang_id}"
    sents = df['sentence'].values
    paths = df['path'].values

    spk_map = create_spk_map(df)

    a = Alphabet(args.lang_id)
    alphabet = a.get_alphabet()

    #init g2p models
    model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    for path, text in tzip(paths[73521:], sents[73521:]):
        client_id = df[df['path']==path]['client_id'].values[0]
        client_id = str(client_id)
        spk_id = spk_map[client_id]
        path = path.replace('.mp3', '.wav')
        utter_id = path.replace('.wav', '')
        utter_id = utter_id.split('_')[-1]
        f = f"{spk_id}-{args.lang_id}-{utter_id}.wav"
        ipath = f"{args.inp_dir}/{args.lang_id}/clips/{path}"
        opath = f"{out_dir}/{f}"
        shutil.copyfile(ipath, opath)

        fname = f.split('.')[0] + '.normalized.txt'
        text_norm = normalize_text(text, args.lang_id, alphabet)
        with open(f'{out_dir}/{fname}', 'w') as fo:
            fo.write(text_norm)

        #Phonemizer
        fname = f.split('.')[0] + '.phn.txt'
        try:
            phonemes = encode_multilang(text_norm, args.lang_id, model, tokenizer)
        except ValueError:
            with open(f'preproc-{args.lang_id}.log', 'a') as fo:
                fo.write(fname+'\n')
            print(f"Skipping file {fname}")
            continue
        with open(f'{out_dir}/{fname}', 'w') as fo:
            fo.write(phonemes)

parser = argparse.ArgumentParser(
                    prog = 'preprocess',
                    description = 'Create CV dataset for TTS input')

parser.add_argument('-i', '--inp_dir', help='Path to Common Voice directory', required=True)
parser.add_argument('-o', '--out_dir', help='Path to store the result')
parser.add_argument('-l', '--lang_id', help='Id of the language to preprocess', required=True)

args = parser.parse_args()
cv_from_validated(args)