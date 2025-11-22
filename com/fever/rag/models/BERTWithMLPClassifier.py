from torch import nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput

class BERTWithMLPClassifier(nn.Module):
    """
    BERT encoder with LoRA adapters + Multi-Layer Perceptron classifier head.

    Architecture:
        BERT (with LoRA) → MLP [768 → 256 → 128 → 2]

    Both LoRA parameters and MLP weights are trainable.
    """

    def __init__(
            self,
            model_name,
            num_labels=2,
            mlp_hidden_dims=[256, 128],
            mlp_dropout=0.3,
            activation='gelu'
    ):
        super(BERTWithMLPClassifier, self).__init__()

        # Load BERT encoder (without default classifier head)
        config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.config = config

        # Select activation function
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        self.activation = activations.get(activation.lower(), nn.GELU())

        # Build MLP classifier head
        layers = []
        input_dim = self.hidden_size

        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation,
                nn.Dropout(mlp_dropout)
            ])
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, num_labels))
        self.classifier = nn.Sequential(*layers)

        # Print architecture
        print(f"\n🏗️  MLP Classifier Architecture:")
        print(f"   Input:  {self.hidden_size} (BERT hidden size)")
        for i, dim in enumerate(mlp_hidden_dims):
            print(
                f"   Layer {i + 1}: Linear({input_dim if i == 0 else mlp_hidden_dims[i - 1]} → {dim}) → {activation.upper()} → Dropout({mlp_dropout})")
        print(f"   Output: Linear({mlp_hidden_dims[-1]} → {num_labels})")

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        """
        Forward pass with loss calculation.

        Args:
            input_ids: Input token IDs (batch_size, seq_length)
            attention_mask: Attention mask (batch_size, seq_length)
            labels: Ground truth labels (batch_size, seq_length)
            inputs_embeds: Optional pre-computed embeddings
            **kwargs: Additional arguments (ignored)

        Returns:
            dict with 'loss' and 'logits'
        """
        # Get BERT encodings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True
        )

        # Token-level hidden states: (batch_size, seq_length, hidden_size)
        sequence_output = outputs.last_hidden_state

        # Pass through MLP classifier: (batch_size, seq_length, num_labels)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # CrossEntropyLoss automatically ignores -100 labels
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
