import torch
import torch.nn as nn


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

    def forward(self, inputs, predictions, direction="to_tgt"):
        # batch = (ids, raw_texts, raw_translations, speakers, texts, src_lens, max_text_lens, mels, mel_lens,
        #          max_mel_lens, translations, translation_lens, max_translation_len, speaker_embeddings, alignments, 
        #          pitches, energies, durations)

        (
            mel_targets,
            _, _, _, _, _, _, _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[7:]
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
            _,
            e,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

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

        # Calculate mel loss only in reverse direction
        if direction == "to_src":
            mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)


        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # Prosody Loss
        # pros_loss = self.pros_loss(e)  # Make sure to account for realignment in foreign language
        # print("Prosody Loss: ", pros_loss, pros_loss.shape) # Should be [Batch, 1] or [Batch]

        # # phone Loss  (Requires extracting predicted phonemes from mel Spectrogram) whisper
        # phone_loss = self.phone_loss()

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
            # + pros_loss + phone_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            # pros_loss,
        )
    
    def pros_loss(self, x, y):
        """
        Calculate the negative log-likelihood of the phone sequence given the prosody features
        Input:
            x: (h_sd, h_si, prev_e)
            y: prosody embeddings e_k from prosody extractor
        Output:
            -loglik: Negative log-likelihood of the phone sequence given the prosody features
        """
        log_pi, mu, sigma = x
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik  # Sum over all phones for total loss (L_pp)

    def phone_loss(self, x, y):
        """
        Calculates the phone loss
        """
        return None

