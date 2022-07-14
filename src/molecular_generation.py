import json
from argparse import ArgumentParser
from transformers import RobertaTokenizer, RobertaForCausalLM
from pathlib import Path


def main(args):
    ckpt = Path(args.model)
    tokenizer = RobertaTokenizer.from_pretrained(str(ckpt))
    model = RobertaForCausalLM.from_pretrained(str(ckpt))
   
    input_ids = tokenizer.encode('', return_tensors='pt')
    args_dict = vars(args)
    generation_params = {k:v for k, v in args_dict.items() if k not in ['model', 'output_file']}
    output = model.generate(input_ids, **generation_params)
    output_decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in output]
    with open(f'predictions/{args.model}.json', 'w') as f:
        f.write(json.dumps({'predictions': output_decoded, 'model': str(ckpt), **generation_params}))
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_return_sequences', type=int, required=True)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--top_p', type=float)
    args = parser.parse_args()
    main(args) 

