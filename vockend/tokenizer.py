
import tensorflow as tf
import tensorflow_text as text
import pathlib
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


from .utils import add_start_end, cleanup_text, load_translation


class TranslationTokenizer():
    def __init__(
        self, name,
        csv_path=None,
        csv_sep=";",
        in_out_reversed=False,
        val_rate=0.1,
        vocab_size=1000,
        bert_tokenizer_params=dict(lower_case=True),
        reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
    ):
        self.name = name
        self.csv_path = csv_path
        self.csv_sep = csv_sep
        self.val_rate = val_rate
        self.in_out_reversed = in_out_reversed
        self.bert_tokenizer_params = bert_tokenizer_params
        self.bert_vocab_args = dict(
            # The target vocabulary size
            vocab_size=vocab_size,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=reserved_tokens,
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params=bert_tokenizer_params,
            # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
            learn_params={},
        )

    def fit(self, data=None):
        train_examples, val_examples = self._load_translation_data(
            data, self.csv_path, self.csv_sep, self.val_rate)

        train_in = train_examples.map(lambda r, _: r)
        train_out = train_examples.map(lambda _, l: l)

        if self.in_out_reversed:
            train_in, train_out = train_out, train_in

        in_vocab, out_vocab = self._bert_vocab(train_in, train_out)
        file_in, file_out = self._save_vocab(in_vocab, out_vocab)

        self.t_in = Tokenizer(
            file_in, self.bert_tokenizer_params, self.bert_vocab_args["reserved_tokens"])
        self.t_out = Tokenizer(
            file_out, self.bert_tokenizer_params, self.bert_vocab_args["reserved_tokens"])
        return train_examples, val_examples, train_in, train_out

    def _save_vocab(self, in_vocab, out_vocab, dir=None):
        dir = dir if dir != None else f"{self.name}_vocab"
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        file_in = f'{dir}/in_vocab.txt'
        file_out = f'{dir}/out_vocab.txt'
        self._write_vocab_file(file_in, in_vocab)
        self._write_vocab_file(file_out, out_vocab)
        return file_in, file_out

    def _load_translation_data(self, data=None, filepath=None, sep=";", val_rate=0.1):
        examples = load_translation(data, filepath, sep, val_rate)
        return examples['train'], examples['validation']

    def _bert_vocab(self, train_in, train_out):
        in_vocab = bert_vocab.bert_vocab_from_dataset(
            train_in.batch(100).prefetch(2),
            **self.bert_vocab_args
        )
        out_vocab = bert_vocab.bert_vocab_from_dataset(
            train_out.batch(100).prefetch(2),
            **self.bert_vocab_args
        )

        return in_vocab, out_vocab

    def _write_vocab_file(self, filepath, vocab):
        with open(filepath, 'w') as f:
            for token in vocab:
                print(token, file=f)


class Tokenizer():
    def __init__(self, vocab_path, bert_tokenizer_params, reserved_tokens):
        self.tokenizer = text.BertTokenizer(
            vocab_path, **bert_tokenizer_params)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = vocab_path

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        enc = enc.merge_dims(-2, -1)
        enc = add_start_end(enc, self._reserved_tokens)
        return enc

    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    def get_vocab_path(self):
        return self._vocab_path

    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)
