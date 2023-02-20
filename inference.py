#Use tts_env for inference

from transformers import T5ForConditionalGeneration, AutoTokenizer
model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
model = model.cuda('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

from ipatok import tokenise as ipa_tokenizer




s = 'Passados mais de cinco dias desde o terremoto de sete e oito de magnitude que atingiu a Turquia e a Síria'

words = s.lower().split()
words = ['<por-bz>'+i for i in words]
out = tokenizer(words,padding=True,add_special_tokens=False,return_tensors='pt')
out = out.to('cuda:0')
preds = model.generate(**out,num_beams=1)
phones = tokenizer.batch_decode(preds.tolist(),skip_special_tokens=True)

sent = ''
for p in phones:
    p = ipa_tokenizer(p)
    p = ' '.join(p)+' _ '
    sent+=p

with open('g2p_inf/eg1.txt', 'w') as fo:
    fo.write(sent[:-3])