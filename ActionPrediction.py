import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class ActionPrediction(BaseModel):
    def __init__(
        self,
        params,
        context_encoder=None,
        pretrained_word_embedders=None,
        Robert_params=None,
        **kwargs
    ):
        super(RobertActionPrediction, self).__init__()

        self.params = params
        self.y_attr = self.params.y_attr
        self.mask_actions = getattr(self.params, "mask_actions", False)
        # for backwards compatibility masking is set to False if state isn't present
        try:
            if self.params.fields["state"]["update_before_and_after_data_point_creation"] < 1:
                self.mask_actions = False
        except LookupError:
            self.mask_actions = False
        logger.info("Action masking set to {}".format(self.mask_actions))
        self.action_vocab = self.params.datafields[self.params.y_attr][1].vocab
        self.action_name_to_id = self.action_vocab.stoi
        self.id_to_action_name = self.action_vocab.itos


        self.feed_forward_layer_dropout = ff_params.get(
            "feed_forward_layer_dropout", DEFAULT_FEED_FORWARD_LAYER_DROPOUT
        )
        self.activation_function_type = ff_params.get(
            "activation_function_type", DEFAULT_ACTIVATION_FUNCTION_TYPE
        )
        self.batch_normalization = ff_params.get("batch_normalization", DEFAULT_BATCH_NORMALIZATION)

        self.ood_index = None
        ood_action = BERT_NLG + "_" + OUT_OF_DOMAIN

        if ood_action in self.action_name_to_id:
            self.ood_index = self.action_name_to_id[ood_action]
        else:
            logger.info("OOD action not present in action vocab!!")

        if hasattr(self.params, "replace_lower_bin_with_ood"):
            if not self.params.replace_lower_bin_with_ood:
                self.ood_index = None

        id_to_action_dict = {idx: action for idx, action in enumerate(self.id_to_action_name)}

        self.metrics = {"accuracy": Accuracy(), "f1_measure": F1Measure(id_to_action_dict)}

        # These thresholds are used for binning the scores of predicted
        # actions. These are currently heuristic based. In future, binning
        # will be done by re-ranker on the combined score of the NLU
        # interpretation

        self.UPPER_BOUND = DEFAULT_OOD_UPPER_BOUND
        if hasattr(self.params, "ood_upper_bound"):
            self.UPPER_BOUND = self.params.ood_upper_bound

        if hasattr(self.params, "ood_lower_bound"):
            self.LOWER_BOUND = self.params.ood_lower_bound
        else:
            self.LOWER_BOUND = max(DEFAULT_MAX_OOD_LOWER_BOUND, float(1 / len(id_to_action_dict)))

        # This threshold is used to discard a site switch action
        self.SITE_SWITCH_THRESHOLD = getattr(
            self.params, "site_switch_threshold", DEFAULT_SITE_SWITCH_THRESHOLD
        )


        # Create intermediate feed-forward layers
        intermediate_feed_forward_layers = []
        # The first dropout applies to the output of context encoders
        if self.feed_forward_layer_dropout > 0:
            intermediate_feed_forward_layers.append(nn.Dropout(self.feed_forward_layer_dropout))
        for i in range(self.num_feed_forward_layers - 1):
            intermediate_feed_forward_layers.append(
                nn.Linear(
                    in_features=self.context_encoder.output_dim,
                    out_features=self.context_encoder.output_dim,
                )
            )
            if self.activation_function_type:
                activation_fcn = getattr(nn, self.activation_function_type, None)
                if activation_fcn:
                    intermediate_feed_forward_layers.append(activation_fcn())
                else:
                    logger.warning(
                        "Invalid activation function type: {}. Skipping "
                        "activation. For valid options see shorturl.at/mwAX2".format(
                            self.activation_function_type
                        )
                    )
            if self.feed_forward_layer_dropout > 0:
                intermediate_feed_forward_layers.append(nn.Dropout(self.feed_forward_layer_dropout))
            if self.batch_normalization:
                intermediate_feed_forward_layers.append(
                    nn.BatchNorm1d(num_features=self.context_encoder.output_dim)
                )
        self.intermediate_feed_forward_layers = (
            nn.Sequential(*intermediate_feed_forward_layers) or None
        )

        self.hidden2label = nn.Linear(
            self.context_encoder.output_dim, params.vocab_sizes[params.y_attr]
        )

    def forward(self, batch):
        # Encode context
        logger.debug("Start ActionPrediction context encoding")
        start_context_encoding = time.time()
        context_embeddings = self.context_encoder(batch)[0]
        end_context_encoding = time.time()
        logger.debug(
            "End ActionPrediction encoding context: {:4.4f} ms".format(
                (end_context_encoding - start_context_encoding) * 1000
            )
        )

        logger.debug("Start ActionPrediction computation")
        start_computation = time.time()
        # remove the sequence length dimension
        output_tensor = context_embeddings[-1, :, :]
        # Feed forward layers after concat all embeddings
        if self.intermediate_feed_forward_layers:
            output_tensor = self.intermediate_feed_forward_layers(output_tensor)
        # Compute logits. shape(logits) = batch_size x target_vocab_size
        logits = self.hidden2label(output_tensor)
        end_computation = time.time()
        logger.debug(
            "End ActionPrediction computation: {:4.4f} ms".format(
                (end_computation - start_computation) * 1000
            )
        )
        if self.mask_actions:
            logits[~self.get_mask(batch)] = float("-inf")
        return logits


    def loss_fn(self, output_scores, labels, sample_weights=None):
        """Computes cross entropy loss between model predicted scores and
        ground-truth labels
        :param output_scores: tensor of shape batch_size x y_vocab_size
        :param labels: tensor of shape 1 x batch_size
        :return: average cross entropy loss, number of examples
        """
        labels = labels.view(-1)
        if sample_weights is None:
            return F.cross_entropy(output_scores, labels)
        else:
            loss = F.cross_entropy(output_scores, labels, reduction="none")
            return sum(loss * sample_weights) / len(loss)
