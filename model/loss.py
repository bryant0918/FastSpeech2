import torch
import torch.nn as nn


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

    def forward(self, inputs, predictions, direction="to_tgt"):
        """
        When going to_tgt everything should be in tgt space.
        When going to_src everything should be in src space.
        """
        (
            mel_targets, 
            pitch_targets, 
            energy_targets, 
            duration_targets,
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
            _,
            e_src,
            e_src_hat,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False

        # print("Src_masks ", src_masks.shape)
        # print("Pitch_predictions ", pitch_predictions.shape)
        # print("is nan pitch_predictions: ", torch.isnan(pitch_predictions).any())
        # print("Pitch_targets ", pitch_targets.shape)

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
        mel_loss, postnet_mel_loss, pros_loss = 0, 0, 0
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

            # mel_duration_loss = self.mel_duration_loss(postnet_mel_predictions, mel_targets)

            pros_loss = self.pros_loss2(e_src_hat, e_src)

            print("Mel Loss: ", mel_loss)
            print("Postnet Mel Loss: ", postnet_mel_loss)
            print("Prosody Loss: ", pros_loss, pros_loss.shape) # Should be [Batch, 1] or [Batch]

            # print("mel_duration_loss: ", mel_duration_loss)

        
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # # word_loss  (Requires extracting predicted phonemes from mel Spectrogram) whisper
        # word_loss = self.word_loss()

        
        print("Pitch Loss: ", pitch_loss)
        print("Energy Loss: ", energy_loss)
        print("Duration Loss: ", duration_loss)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + pros_loss 
            # + word_loss
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
        print("log_pi shape: ", log_pi.shape)   # log_pi shape:  torch.Size([2, 112, 8])
        print("mu shape: ", mu.shape)           # mu shape:  torch.Size([2, 112, 8, 256])
        print("sigma shape: ", sigma.shape)     # sigma shape:  torch.Size([2, 112, 8, 256])
        print("y shape (extracted): ", y.shape) # y shape:  torch.Size([2, 80, 807, 256])

        # TODO: Somehow map y from [Batch, melspec H, melspec W, 256] to [Batch, tgt_seq_len, 256]?


        z_score = (y.unsqueeze(1) - mu) / sigma # Value Error Here
        normal_loglik = (
                -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
                - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik  # Sum over all phones for total loss (L_pp)

    def pros_loss2(self, x, y):
        import torch.distributions as dist

        log_pi, mu, sigma = x
        n_batches, n_samples, n_components, n_features = mu.shape
        n_components = log_pi.shape[-1]
        
        # Initialize the log likelihoods for each batch
        log_likelihoods = torch.zeros((n_batches, n_samples, n_components))

        print("sigma has negative vals", (sigma <= 0).any())
        print("Shape of sigma[b,:,k]: ", sigma[0,:,0].shape) 
        #mvn wants this to be square but it's [58, 58, 256]
        #Do I need a linear layer here or something to make it [58,58]

        print("log_pi shape: ", log_pi.shape)   # log_pi shape:  torch.Size([2, 112, 8])
        print("mu shape: ", mu.shape)           # mu shape:  torch.Size([2, 112, 8, 256])
        print("sigma shape: ", sigma.shape)     # sigma shape:  torch.Size([2, 112, 8, 256])
        print("y shape (extracted): ", y.shape) # y shape:  torch.Size([2, 80, 807, 256])
        
        print("Diag sigma", torch.diagonal(sigma[0,:,0]).shape))
        
        # This needs to be per phone as well.
        for b in range(n_batches):
            for k in range(n_samples):
                for i in range(n_components):
                    # Create a multivariate normal distribution for the k-th component
                    mvn = dist.MultivariateNormal(loc=mu[b,k,i], covariance_matrix=sigma[b,k,i])
                    
                    # Compute the log likelihood for each point in the batch for the k-th component
                    log_likelihoods[b, k, i] = mvn.log_prob(y)
        
        # Compute the log of the weighted sum of the probabilities
        weighted_log_likelihoods = log_likelihoods + log_pi
        log_sum_exp = torch.logsumexp(weighted_log_likelihoods, dim=2)
        
        # Compute the negative log-likelihood for each batch
        nlls = -torch.sum(log_sum_exp, dim=1)
        
        return nlls

    def word_loss(self, x, y):
        """
        Calculates the phone loss
        """
        return None

    def mel_duration_loss(self, x, y):
        """
        Calculate the mel duration loss
        x: predicted mel spectrogram
        y: target mel spectrogram
        """

        x_durs = torch.tensor([len(mel) for mel in x])
        y_durs = torch.tensor([len(mel) for mel in y])

        return self.mae_loss(x_durs, y_durs)
    

