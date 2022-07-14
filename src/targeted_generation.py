import json
from tqdm import tqdm
from transformers import EncoderDecoderModel, T5Config
from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from models.t5 import T5ForConditionalGeneration
from data import ProteinTokenizer, ProteinBPETokenizer
from data import SmilesTokenizer, SmilesBPETokenizer
from evaluation import calc_val_metrics
import logging
import time
logger = logging.getLogger(__name__)


def decoding_params(decoding_method):
    params = {'max_length': 128, 'num_return_sequences': 20}
    if decoding_method == 'beam':
        specific_params = {'num_beams': 20, 'early_stopping': True}
    elif decoding_method == 'sampling':
        specific_params = {'do_sample': True, 'top_k': 0}
    elif decoding_method == 'temperature':
        specific_params = {'do_sample': True, 'top_k': 0, 'temperature': 0.9}
    elif decoding_method == 'top_k':
        specific_params = {'do_sample': True, 'top_k': 50}
    elif decoding_method == 'top_p':
        specific_params = {'do_sample': True, 'top_k': 0, 'top_p': 0.9}
    elif decoding_method == 'top_k_p':
        specific_params = {'do_sample': True, 'top_k': 50, 'top_p': 0.95}
    params.update(specific_params)
    return params


def generate_molecule(batch, generation_params):
    # cut off at BERT max length 512
    inputs = src_tokenizer(batch["target_sequence"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(input_ids, attention_mask=attention_mask,
                             decoder_start_token_id=tgt_tokenizer.bos_token_id, eos_token_id=tgt_tokenizer.eos_token_id,
                             pad_token_id=tgt_tokenizer.eos_token_id, **generation_params)

    try:
        output_str = tgt_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except:
        output_str = ['Invalid'] * len(batch)
    return output_str


def generate_for_dataset(data, decoding_methods=['beam', 'greedy', 'top_k', 'top_p'], path=None):
    predictions = {}
    for method in decoding_methods:
        start_time = time.time()
        params = decoding_params(method)
        logger.info(f'Decoding with {method} {params}')
        predictions[method] = {**params}
        decoded_molecules = []
        for item in tqdm(data['train']):
            molecules = generate_molecule(item, params)
            predictions[method][item['UniProt_S_ID']] = molecules
            decoded_molecules.extend(molecules)
        predictions[method].update(calc_val_metrics(decoded_molecules))
        logger.info(f'Time:{time.time()-start_time}')
    with open(path, 'w') as f:
        f.write(json.dumps(predictions))


parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--protein_tokenizer', type=str, default=None)
parser.add_argument('--ligand_tokenizer', type=str, default=None)
parser.add_argument('--test', type=str, default='data/splits/test_interactions_uniq.csv')
parser.add_argument('--val', type=str, default='data/splits/val_interactions_uniq.csv')
parser.add_argument('--decoding', type=str, default='sampling')


args = parser.parse_args()
model_dir = Path(args.model)
logger.info(f'{args}')

if args.protein_tokenizer == 'char':
    src_tokenizer = ProteinTokenizer()
else:
    src_tokenizer = ProteinBPETokenizer()

logging.info(f'Loading ligand tokenizer {args.ligand_tokenizer}')

if args.ligand_tokenizer == 'char':
    tgt_tokenizer = SmilesTokenizer()
else:
    tgt_tokenizer = SmilesBPETokenizer()


if 't5' in str(model_dir):
    config = T5Config.from_pretrained(model_dir)
    config.tie_word_embeddings = False
    config.max_length = 128
    config.bos_token_id = tgt_tokenizer.bos_token_id
    config.eos_token_id = tgt_tokenizer.eos_token_id
    model = T5ForConditionalGeneration.from_pretrained(model_dir, src_vocab_size=src_tokenizer.vocab_size,
                                                       tgt_vocab_size=tgt_tokenizer.vocab_size, config=config)
else:
    model = EncoderDecoderModel.from_pretrained(model_dir)


pred_dir = Path('predictions') / args.model
pred_dir.mkdir(parents=True, exist_ok=True)

batch_size = 1  # change to 64 for full evaluation
test_data = load_dataset('csv', data_files=[args.test])
val_data = load_dataset('csv', data_files=[args.val])

generate_for_dataset(test_data, args.decoding.split(','), path=pred_dir / 'test_predictions.json')
generate_for_dataset(val_data, args.decoding.split(','), path=pred_dir / 'val_predictions.json')

