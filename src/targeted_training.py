import time
import logging
import json
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import EncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import T5Config
from data import ProteinTokenizer, ProteinBPETokenizer
from data import SmilesTokenizer, SmilesBPETokenizer
from models.t5 import T5ForConditionalGeneration
from evaluation import calc_val_metrics
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_PATHS = {
    "ChemBERTaLM": "models/ChemBERTaLM",
    "ChemBERTa": "seyonec/PubChem10M_SMILES_BPE_450k",
    "ProteinRoBERTa": "models/ProteinRoBERTa",
}


def compute_metrics(pred):
    pred_ids = pred.predictions
    # all unnecessary tokens are removed
    pred_str = tgt_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pred_tokens = [item.split(' ') for item in pred_str]
    preds = [''.join(item) for item in pred_tokens]

    path = Path(f'{args.output_dir}/predictions')
    path.mkdir(exist_ok=True, parents=True)
    with open(path / f'{time.strftime("%Y%m%d-%H%M%S")}', 'w') as f:
        f.write(json.dumps(preds))
    return calc_val_metrics(preds)


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = src_tokenizer.batch_encode_plus(batch["target_sequence"], padding="max_length",
                                             truncation=True, max_length=args.encoder_max_length)
    outputs = tgt_tokenizer.batch_encode_plus(batch["canonical_SMILES"], padding="max_length",
                                              truncation=True, max_length=args.decoder_max_length)

    batch["input_ids"] = inputs['input_ids']
    batch["attention_mask"] = inputs['attention_mask']
    batch["decoder_input_ids"] = outputs['input_ids']
    batch["decoder_attention_mask"] = outputs['attention_mask']
    batch["labels"] = outputs.input_ids.copy()

    # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
    # We have to make sure that the PAD token is ignored
    # Lastly, it is very important to remember to ignore the loss of the padded labels. In 🤗Transformers this can be done by setting the label to -100.

    batch["labels"] = [[-100 if token == tgt_tokenizer.pad_token_id else token for token in labels] for labels in
                       batch["labels"]]

    return batch


def build_encoder_decoder_model(protein_model, ligand_model):

    model = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_PATHS[protein_model], MODEL_PATHS[ligand_model])
    model.config.max_length = 128
    model.config.bos_token_id = tgt_tokenizer.bos_token_id
    model.config.eos_token_id = tgt_tokenizer.eos_token_id
    return model


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

parser = ArgumentParser()
parser.add_argument('--protein_model', type=str)
parser.add_argument('--ligand_model', type=str)
parser.add_argument('--model_name_or_path', type=str, default=None)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_train_epochs', type=int, default=3)
parser.add_argument('--protein_tokenizer', type=str, default=None)
parser.add_argument('--ligand_tokenizer', type=str, default=None)
parser.add_argument('--train', type=str, default=None)
parser.add_argument('--val', type=str, default=None)
parser.add_argument('--max_train_samples', type=int, default=None)
parser.add_argument('--max_val_samples', type=int, default=None)
parser.add_argument('--freeze', type=str, default=None)
parser.add_argument('--t5_model', type=str, default='t5_tiny')
parser.add_argument('--warmup_steps', type=int, default=2000)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--encoder_max_length', type=int, default=512)
parser.add_argument('--decoder_max_length', type=int, default=128)

args = parser.parse_args()
# train_data, val_data, protein_model, ligand_model, output_dir
logging.info(f'Loading train dataset {args.train}')
train_dataset = load_dataset('csv', data_files=[args.train])
logging.info(f'Loading train dataset {args.val}')
val_dataset = load_dataset('csv', data_files=[args.val])

logging.info(f'Loading protein tokenizer {args.protein_tokenizer}')

if args.protein_tokenizer == 'char':
    src_tokenizer = ProteinTokenizer()
else:
    src_tokenizer = ProteinBPETokenizer()

logging.info(f'Loading ligand tokenizer {args.ligand_tokenizer}')

if args.ligand_tokenizer == 'char':
    tgt_tokenizer = SmilesTokenizer()
else:
    tgt_tokenizer = SmilesBPETokenizer()

if args.model_name_or_path:
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, src_vocab_size=src_tokenizer.vocab_size,
                                       tgt_vocab_size=tgt_tokenizer.vocab_size)
    learning_rate = 5e-05
elif args.protein_model is not None and args.ligand_model is not None:
    model = build_encoder_decoder_model(args.protein_model, args.ligand_model)
    learning_rate = 5e-05
else:
    config = T5Config.from_pretrained(f'{args.t5_model}')
    config.tie_word_embeddings = False
    config.max_length = 128
    config.bos_token_id = tgt_tokenizer.bos_token_id
    config.eos_token_id = tgt_tokenizer.eos_token_id
    model = T5ForConditionalGeneration(config=config, src_vocab_size=src_tokenizer.vocab_size,
                                       tgt_vocab_size=tgt_tokenizer.vocab_size)
    learning_rate = 0.0001


logging.info(f'Loaded model {model.config.model_type}')

train_data = train_dataset['train'].map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=args.batch_size,
    remove_columns=["target_sequence", "canonical_SMILES", "UniProt_S_ID"]
)

if args.max_train_samples:
    train_data = train_data.select(range(args.max_train_samples))
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_data = val_dataset['train'].map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=args.batch_size,
    remove_columns=["target_sequence", "canonical_SMILES", "UniProt_S_ID"]
)
if args.max_val_samples:
    val_data = val_data.select(range(args.max_val_samples))
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=False,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    output_dir=args.output_dir,
    warmup_steps=args.warmup_steps,
    learning_rate=learning_rate,  # for t5
    save_total_limit=5,
    num_train_epochs=args.num_train_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    load_best_model_at_end=True,
    # report_to='wandb',
    no_cuda=True,
    run_name=args.output_dir
)

logging.info(f'Training with following arguments {training_args.to_json_string()}')

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tgt_tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data
)
trainer.train()

model.save_pretrained(args.output_dir)
