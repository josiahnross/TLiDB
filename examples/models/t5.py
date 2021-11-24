from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

from .TLiDB_model import TLiDB_model

class T5(TLiDB_model):
    _encoder_only = False
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = T5Tokenizer.from_pretrained(config.model)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model)
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        self.lm_head = self.model.lm_head
        self.layers = [self.encoder, self.decoder, self.lm_head]                
    
    # def _forward(self, inputs, y_true=None):
    def _forward(
        self,
        input_ids = None,
        attention_mask = None,
        encoder_outputs = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        lm_labels = None
    ):
        if encoder_outputs == None:
            encoder_outputs = self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state

        if lm_labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(lm_labels)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                        attention_mask=decoder_attention_mask,
                                        encoder_hidden_states=encoder_outputs,
                                        encoder_attention_mask=attention_mask
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model.model_dim ** -0.5) # unclear whether this is beneficial
        lm_logits = self.lm_head(sequence_output)

        loss = None
        y_true = None

        if lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            y_true = self.tokenizer.batch_decode(lm_labels, skip_special_tokens=True)

        pred_tokens = self.decode_logits(lm_logits)

        return pred_tokens, loss, y_true

    def transform_inputs(self, inputs):
        """Only tokenizes inputs"""
        tokenized_inputs = self.tokenizer(inputs, padding="max_length",truncation=True, max_length=self.config.max_seq_length, return_tensors="pt")
        return tokenized_inputs

    def transform_outputs(self, outputs, task,dataset_name):
        """tokenizes outputs"""
        tokenized_outputs = self.tokenizer(outputs, padding="longest", truncation=True, return_tensors="pt")
        return tokenized_outputs.input_ids

    def decode_logits(self, logits):
        assert logits.dim() > 1
        pred_tokens = logits.argmax(-1)
        return self.tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)