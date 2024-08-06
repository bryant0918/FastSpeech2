import torch
import torch.nn as nn
import torch.distributions as dist
from transformers import BertModel, BertTokenizer
# import whisper
from whisperX import whisperx
from text import id_to_lang


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config, train_config, device='cpu'):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.pros_loss = ProsLoss()
        self.word_loss = WordLoss(model_config, device)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.pros_weight = train_config['loss']['pros_weight']
        self.pitch_energy_weight = train_config['loss']['pitch_energy_weight']
        self.full_duration_weight = train_config['loss']['full_duration_weight']
        self.label_smoothing = train_config['loss']['label_smoothing']

        self.batch_size = train_config['optimizer']['batch_size']

    def forward(self, inputs, predictions, direction="to_tgt", word_step=10):
        """
        When going to_tgt everything should be in tgt space.
        When going to_src everything should be in src space.
        """
        (   text,
            mel_targets, 
            mel_lens_targets,
            lang_targets,
            pitch_targets, 
            energy_targets, 
            log_duration_targets,
        ) = inputs

        (   mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            mel_lens_predictions,
            extracted_e,
            predicted_e,
            prosody_reg_term,
            npc_loss,
            audio,
            pred_generated,
        ) = predictions
        device = mel_masks.device
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        mel_lens_targets.requires_grad = False
        # Stop Gradient for targets
        log_duration_targets = log_duration_targets.detach()
        pitch_targets = pitch_targets.detach()
        energy_targets = energy_targets.detach()

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        
        # Calculate mel loss and prosody loss only in reverse direction
        mel_loss, postnet_mel_loss, pros_loss = torch.tensor([0]).to(device), torch.tensor([0]).to(device), torch.tensor([0]).to(device)
        if direction == "to_src":
            mel_targets = mel_targets[:, : mel_masks.shape[1], :]
            mel_targets.requires_grad = False

            mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

            # Penalize later parts of sequence more:
            # position_weights = torch.linspace(0.25, 1, mel_masks.shape[1]).unsqueeze(-1).to(device).requires_grad_(False)
            # position_weights /= torch.sum(position_weights)

            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)


            # Multiply by position weights and sum to get scalar loss
            # mel_loss = torch.sum(mel_loss * position_weights)
            # postnet_mel_loss = torch.sum(postnet_mel_loss * position_weights)

            if extracted_e is not None:
                extracted_e = extracted_e.detach()
                # TODO: Figure out best beta value
                pros_loss = self.pros_loss(predicted_e, extracted_e, src_masks[:self.batch_size//2]) * self.pros_weight

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        pitch_loss = pitch_loss * self.pitch_energy_weight
        energy_loss = energy_loss * self.pitch_energy_weight
        
        # word_loss  (Requires extracting predicted phonemes from mel Spectrogram) whisper
        if audio is not None:
            word_loss = self.word_loss(audio, text, lang_targets)
        else:
            word_loss = torch.tensor([0]).to(device)

        # Full Duration Loss
        full_duration_loss = self.mae_loss(mel_lens_predictions, mel_lens_targets.float()) * self.full_duration_weight

        # Calculate discriminator loss for generator with label smoothing
        g_loss = self.bce_loss(pred_generated, torch.ones_like(pred_generated) * self.label_smoothing)

        total_loss = (mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss 
                      + pros_loss + word_loss + full_duration_loss + g_loss + prosody_reg_term + npc_loss
        )
        
        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            pros_loss,
            word_loss,
            full_duration_loss,
            g_loss,
            prosody_reg_term,
            npc_loss,
        )


class ProsLoss(nn.Module):
    def __init__(self):
        super(ProsLoss, self).__init__()

    def forward(self, x, y, src_masks):
        """
        Calculate the negative log-likelihood of the phone sequence given the prosody features
        Input:
            x: (log_pi, mu, sigma) from prosody predictor
            y: prosody embeddings e_k from prosody extractor
            src_masks: Source masks
        Output:
            Negative log-likelihood of the phone sequence given the prosody features
        """
        log_pi, mu, sigma = x
        # print()
        # sigma = torch.sqrt(sigma)

        # print("sum log_pi", torch.sum(torch.exp(log_pi)).item(), log_pi.shape) # Checks out (sums to 1)

        # print("Log_pi", torch.min(log_pi).item(), torch.max(log_pi).item())  # Range: (-inf, 0)
        # print("Mu", torch.min(mu).item(), torch.max(mu).item())              # Range: (-inf, inf)
        # print("Sigma", torch.min(sigma).item(), torch.max(sigma).item())     # Range: (0, inf)

        batch_size, max_seq_len = mu.shape[0], mu.shape[1]

        # print("y shape", y.shape) # Shape torch.Size([2, 83, 256])
        # print("mu shape", mu.shape) # Shape torch.Size([2, 83, 8, 256])
        # print("sigma shape", sigma.shape) # Shape torch.Size([2, 83, 8, 256])

        z_score = (y.unsqueeze(2) - mu) / torch.sqrt(sigma)                # torch.Size([2, seq, 8, 256])
        # print("z_score", torch.min(z_score).item(), torch.max(z_score).item()) # Range: (-inf, inf)

        # print("einsum", torch.min(torch.einsum("bkih,bkih->bki", z_score, z_score)).item()) # Should be positive
        # print("sum of log", torch.max(torch.sum(torch.log(sigma), dim=-1)).item())  # Should be negative

        # normal_loglik = (-0.5 * torch.einsum("bkih,bkih->bki", z_score, z_score)  # Should be negative
        #                  - torch.sum(torch.log(sigma), dim=-1))  # torch.Size([2, seq, 256])
        
        # print("normal_loglik", torch.min(normal_loglik).item(), torch.max(normal_loglik).item()) # Range: (-inf, 0)
        # loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1) # torch.Size([2, seq] # Should be negative
        # print("Loglik", torch.min(loglik).item(), torch.max(loglik).item()) # Range: (-inf, 0)
        
        # q = torch.einsum("bkih,bkih->bki", z_score, z_score) # q is too small at 1.2.
        # print("q", torch.min(q).item(), torch.max(q).item()) # Range: (0, inf)
        # eq = torch.exp(-0.5 * q)  # Too big at .5 if being divided by small number
        # print("eq", torch.min(eq).item(), torch.max(eq).item(), eq.shape) # Range: (0, 1)

        # sum_sigma = torch.sum(sigma, dim=-1)
        # print("sum_sigma", torch.min(sum_sigma).item(), torch.max(sum_sigma).item()) # Range: (0, inf)

        # normal_loglik = eq / sum_sigma
        # print("normal_loglik", torch.min(normal_loglik).item(), torch.max(normal_loglik).item()) # Range: (0, 1) but more realistically between (0.1, 0.3)

        normal_loglik = (torch.exp(-0.5 * torch.einsum("bkih,bkih->bki", z_score, z_score))
                        / torch.sum(sigma, dim=-1))  # torch.Size([2, seq, 8])
        
        if torch.isnan(normal_loglik).any() or torch.isinf(normal_loglik).any():
            print("Normal Loglik contains NaNs or infs")
            print(normal_loglik)
            raise ValueError
        
        if torch.isnan(log_pi).any() or torch.isinf(log_pi).any():
            print("log_pi contains NaNs or infs")
            print(log_pi)
            raise ValueError
        # normal_loglik = 1/torch.sqrt(torch.sum(sigma, dim=-1)) * torch.exp(-0.5 * ((y.unsqueeze(2) - mu)**2 / sigma))

        # print("normal_loglik", torch.min(normal_loglik).item(), torch.max(normal_loglik).item(), normal_loglik.shape) # Range: (-inf, 0)
        
        # print("shape of log_pi*normal_loglike", (torch.exp(log_pi)*normal_loglik).shape) # Shape
        
        # Getting RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
        # pi_normal_loglik = torch.exp(log_pi)*normal_loglik  # Range (0, 1)
        # print("pi_normal_loglik", torch.min(pi_normal_loglik).item(), torch.max(pi_normal_loglik).item())

        # unclamped_likelihood = torch.sum(torch.exp(log_pi)*normal_loglik, dim=-1)
        # print("unclamped_likelihood", torch.min(unclamped_likelihood).item(), torch.max(unclamped_likelihood).item(), unclamped_likelihood.shape)
        # likelihood = torch.clamp(unclamped_likelihood, min=1e-10, max=1-1e-10) # Range (1e-10, 1)
        # print("likelihood", torch.min(likelihood).item(), torch.max(likelihood).item(), likelihood.shape) # Range: (0, 1)

        # Leads to negative loss when not clamping at max value of 1.
        loglik = torch.log(torch.clamp(torch.sum(torch.exp(log_pi)*normal_loglik, dim=-1), min=1e-10, max=1-1e-15)) # Range: (-1, 0)

        # loglik = torch.logsumexp(log_pi + torch.log(normal_loglik), dim=-1)
        # print("Loglik", torch.min(loglik).item(), torch.max(loglik).item(), loglik.shape) # Range: (-1, 0)

        negloglik = -loglik.masked_select(src_masks)

        # print("negloglik", negloglik.shape) # Shape
        # print("Negloglik", torch.min(negloglik).item(), torch.max(negloglik).item()) # Range: (0, 1)

        nlls = torch.sum(negloglik)/max_seq_len 

        return nlls / batch_size


class WordLoss(nn.Module):
    def __init__(self, config, device='cpu'):
        super(WordLoss, self).__init__()
        bert = config['sentence_embedder']['model']
        # self.transcriber = whisper.load_model(config['transcriber']['model'])
        print("DEVICE: ", device)
        self.transcriber = whisperx.load_model(config['transcriber']['model'], device=str(device), compute_type='float16')
        self.tokenizer = BertTokenizer.from_pretrained(bert)
        self.bert_model = BertModel.from_pretrained(bert)
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, audio, text, lang_ids):
        """
        Calculate the word loss.
        Input:
            audio: Audio waveform
            text (str): Raw Text
            lang_ids: Language ids
        Performs Speech to Text on audio and compares to text.
        Uses 
        Output: Loss
        """
        pred_texts = []

        import time
        start = time.time()
        for i, aud in enumerate(audio):
            if aud is None:
                pred_texts.append("")
            else:
                # Language here is id not string
                predicted_text = self.transcriber.transcribe(aud, language=id_to_lang[lang_ids[i].item()])
                if len(predicted_text["segments"]) == 0:
                    pred_texts.append("")
                else:
                    pred_texts.append(predicted_text["segments"][0]['text'])
        print("Time taken to transcribe: ", time.time() - start)

        start = time.time()
        # Tokenize both predicted and reference text
        predicted_tokens = self.tokenizer(pred_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        reference_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        print("Time taken to tokenize: ", time.time() - start)

        start = time.time()
        # Move tokens to the specified device (CUDA if available)
        predicted_tokens = {k: v.to(self.bert_model.device) for k, v in predicted_tokens.items()}
        reference_tokens = {k: v.to(self.bert_model.device) for k, v in reference_tokens.items()}
        print("Time taken to move to device: ", time.time() - start)

        start = time.time()
        # Get BERT embeddings
        with torch.no_grad():  # No need to calculate gradients here
            predicted_embeddings = self.bert_model(**predicted_tokens).last_hidden_state
            reference_embeddings = self.bert_model(**reference_tokens).last_hidden_state
        print("Time taken to get embeddings: ", time.time() - start)
              
        start = time.time()
        # Average the embeddings across the sequence length dimension
        predicted_embeddings_avg = predicted_embeddings.mean(dim=1)
        reference_embeddings_avg = reference_embeddings.mean(dim=1)
        # Ensure the target tensor is on the same device and has the correct shape
        target_tensor = torch.tensor([1]).to(predicted_embeddings.device).expand_as(predicted_embeddings_avg[:, 0])

        loss = self.cosine_loss(predicted_embeddings_avg, reference_embeddings_avg, target_tensor)
        print("Time taken to calculate loss: ", time.time() - start)
        return loss


