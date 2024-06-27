import torch
import torch.nn as nn
import torch.distributions as dist
from transformers import BertModel, BertTokenizer
import whisper


class ProsLearnerLoss(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(ProsLearnerLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        

        return None
    

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
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

        self.word_loss = WordLoss(model_config)

    def forward(self, inputs, predictions, direction="to_tgt"):
        """
        When going to_tgt everything should be in tgt space.
        When going to_src everything should be in src space.
        """
        (
            text,
            mel_targets, 
            mel_lens_targets,
            pitch_targets, 
            energy_targets, 
            log_duration_targets,
        ) = inputs

        (
            mel_predictions,
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
            audio,
        ) = predictions
        device = mel_masks.device
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_lens_targets.requires_grad = False
        
        # Stop Gradient
        extracted_e = extracted_e.detach()

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

            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

            # TODO: Figure out best beta value
            beta = .2
            pros_loss = self.pros_loss(predicted_e, extracted_e, src_masks)*beta

            # print("Mel Loss: ", mel_loss)
            # print("Postnet Mel Loss: ", postnet_mel_loss)
            # print("Prosody Loss: ", pros_loss, pros_loss.shape) # Should be [Batch, 1] or [Batch]

            # print("mel_duration_loss: ", mel_duration_loss)

        
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        
        # word_loss  (Requires extracting predicted phonemes from mel Spectrogram) whisper
        if audio is not None:
            word_loss = self.word_loss(audio, text)
        else:
            word_loss = torch.tensor([0]).to(device)

        # Full Duration Loss
        full_duration_loss = self.mae_loss(mel_lens_predictions, mel_lens_targets.float())

        # print("Pitch Loss: ", pitch_loss)
        # print("Energy Loss: ", energy_loss)
        # print("Duration Loss: ", duration_loss)
        # print("Full Duration Loss: ", full_duration_loss) # Will be 0 during pretraining.

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss 
            + pros_loss + word_loss + full_duration_loss
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
        )


class ProsLoss(nn.Module):
    def __init__(self):
        super(ProsLoss, self).__init__()

    def forward(self, x, y, src_masks):
        """
        Calculate the negative log-likelihood of the phone sequence given the prosody features
        Input:
            x: (h_sd, h_si, prev_e)
            y: prosody embeddings e_k from prosody extractor
        Output:
            Negative log-likelihood of the phone sequence given the prosody features
        """
        log_pi, mu, sigma = x
        sigma = sigma + 1e-8
        batch_size = mu.shape[0]

        z_score = (y.unsqueeze(2) - mu) / sigma                 # torch.Size([2, seq, 8, 256])
        normal_loglik = (-0.5 * torch.einsum("bkih,bkih->bki", z_score, z_score)
                         - torch.sum(torch.log(sigma), dim=-1))  # torch.Size([2, seq, 256])
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1) # torch.Size([2, seq]
        negloglik = -loglik.masked_select(src_masks)
        nlls = torch.sum(negloglik)

        return nlls / batch_size

    def pros_loss2(self, x, y):
        """
        ERROR: Gives wrong results (log_sum_exp) should be probabilities
        Calculate the negative log-likelihood of the phone sequence given the prosody features
        Input:
            x: (h_sd, h_si, prev_e)
            y: prosody embeddings e_k from prosody extractor
        Output:
            -loglik: Negative log-likelihood of the phone sequence given the prosody features
        """

        log_pi, mu, sigma = x
        n_batches, n_samples, n_components, n_features = mu.shape
        n_components = log_pi.shape[-1]
        
        # Initialize the log likelihoods for each batch
        log_likelihoods = torch.zeros((n_batches, n_samples, n_components)).to(log_pi.device)
        
        # This needs to be per phone as well.
        for b in range(n_batches):
            for k in range(n_samples):
                for i in range(n_components):
                    # Create a multivariate normal distribution for the k-th component
                    mvn = dist.MultivariateNormal(loc=mu[b,k,i], covariance_matrix=torch.diag(sigma[b,k,i]))

                    if torch.isnan(y[b,k]).any():
                        continue
                    
                    # Compute the log likelihood for each point in the batch for the k-th component
                    log_likelihoods[b, k, i] = mvn.log_prob(y[b,k])
        
        # Compute the log of the weighted sum of the probabilities
        weighted_log_likelihoods = log_likelihoods + log_pi
        print("weighted_log_likelihoods shape: ", weighted_log_likelihoods.shape) # weighted_log_likelihoods shape:  torch.Size([2, seq, 8]
        
        log_sum_exp = torch.logsumexp(weighted_log_likelihoods, dim=2)
        print("Log_sum_exp shape: ", log_sum_exp.shape) # Log_sum_exp shape:  torch.Size([2, seq])
        print(log_sum_exp)
        # Compute the negative log-likelihood for each batch
        nlls = -torch.sum(log_sum_exp, dim=1)
        print("nlls:", nlls)
        return torch.mean(nlls)


class WordLoss(nn.Module):
    def __init__(self, config):
        super(WordLoss, self).__init__()
        bert = config['sentence_embedder']['model']
        self.transcriber = whisper.load_model(config['transcriber']['model'])
        self.tokenizer = BertTokenizer.from_pretrained(bert)
        self.bert_model = BertModel.from_pretrained(bert)
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, audio, text):
        """
        Calculate the word loss.
        Input:
            audio: Audio waveform
            text (str): Raw Text
        Performs Speech to Text on audio and compares to text.
        Uses 
        
        Output: Loss
        
        """
        pred_texts = []

        for aud in audio:
            predicted_text = self.transcriber.transcribe(aud)
            pred_texts.append(predicted_text['text'])

        # Tokenize both predicted and reference text
        predicted_tokens = self.tokenizer(pred_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        reference_tokens = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # Move tokens to the specified device (CUDA if available)
        predicted_tokens = {k: v.to(self.bert_model.device) for k, v in predicted_tokens.items()}
        reference_tokens = {k: v.to(self.bert_model.device) for k, v in reference_tokens.items()}

        # Get BERT embeddings
        with torch.no_grad():  # No need to calculate gradients here
            predicted_embeddings = self.bert_model(**predicted_tokens).last_hidden_state
            reference_embeddings = self.bert_model(**reference_tokens).last_hidden_state

        # Average the embeddings across the sequence length dimension
        predicted_embeddings_avg = predicted_embeddings.mean(dim=1)
        reference_embeddings_avg = reference_embeddings.mean(dim=1)
        # Ensure the target tensor is on the same device and has the correct shape
        target_tensor = torch.tensor([1]).to(predicted_embeddings.device).expand_as(predicted_embeddings_avg[:, 0])

        loss = self.cosine_loss(predicted_embeddings_avg, reference_embeddings_avg, target_tensor)

        return loss


