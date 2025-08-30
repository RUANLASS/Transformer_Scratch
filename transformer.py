import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super(PositionalEncoding, self).__init__() 
        pos_enc = torch.zeros(max_sequence_length, d_model)
        pos = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1) #0,1,2,3,4... assignment of "i" to the tokens 
        den = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(1000.0)/d_model)) #denominator term of positional encoding, 10000^(2i/d_model)
        #               ^----creating the 2i -------------^    
        #     ^------------ e^(-(2i*log1000)/d_model) = 1/1000^(2i/d_model)------------------^
        pos_enc[:,0::2] = torch.sin(pos*den) #for 0,2,4,6,8,10... or the 2i rows
        pos_enc[:,1::2] = torch.cos(pos*den) #for 1,3,5,7,9.... or the 2i+1 rows

        self.register_buffer('pos_enc', pos_enc.unsqueeze(0)) #tells Pytorch this is non-learnable

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model%num_heads==0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_prod_attention(self, Q, K, V, mask = None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(self.d_k) #only need to transpose the last two dimensions of K
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # masked_fill(condition, value_to_fill)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output
    
    def split_heads(self, input):
        batch_size, seq_length, d_model = input.size()
        return input.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) ##changing up the dimensions from b_s, seq_len, d_model to b_s, seq_len, num_heads, d_k. Transposing seq_len, num_heads

    def concat_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        #                       ^-concat-ing-^

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attention_output = self.scaled_dot_prod_attention(Q, K, V, mask)
        
        output = self.W_o(self.concat_heads(attention_output))
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        A1 = self.relu(self.l1(x))
        return self.l2(A1)

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Encoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(Decoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x





           
class Transformer(nn.Module):   ## inheritance from nn.Module to register the model with PyTorch, offers a lot of functionality
    def __init__(self,enc_vocab_size, dec_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()     ## formality for nn.Module
        self.encoder_embedding = nn.Embedding(enc_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(dec_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, dec_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    
    
    def mask(self, enc, dec): #attention score matrix: [batch_size, num_heads, query_sequence_length, key_sequence_length]
        '''
            enc_mask, dec_mask(initial) = converts input tensors to boolean tensors, padded values (0s) become False, and don't affect the attention 
            no_peeking_mask = ensures that in the future token prediction mechanism, only PAST tokens are taken into account and not future ones
            dec_mask(final) = adds no_peeking functionality to removal of padded values 
        '''
        enc_mask = (enc!=0).unsqueeze(1).unsqueeze(2) # mask is already of shape [batch_size, key_sequence_length], pad the other two dimensions
        dec_mask = (dec!=0).unsqueeze(1).unsqueeze(3) # mask is already of shape [batch_size, query_sequence_length], pad the other two dimensions
        seq_length = dec.size(1)
        no_peeking_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool() # just a unit lower triangular matrix
        dec_mask = no_peeking_mask & dec_mask
        return enc_mask, dec_mask
    
    def forward(self, enc, dec):
        #mask
        enc_mask, dec_mask = self.mask(enc, dec)

        #embedding, encoding dropout
        enc_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(enc)))
        dec_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(dec)))

        enc_output = enc_embedded

        #multihead attention
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, enc_mask)

        #multihead attention
        dec_output = dec_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, enc_mask, dec_mask)

        #passing through regular dense layer
        output = self.fc(dec_output)
        return output


# Generate random sample data
src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    
#training
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

'''transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")'''

#evaluation
#transformer.eval()

# Generate random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

with torch.no_grad():

    val_output = transformer(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")


def calculate_accuracy(output, target, ignore_index=0):
    """Calculate accuracy ignoring padding tokens"""
    mask = target != ignore_index
    predictions = output.argmax(dim=-1)
    correct = (predictions == target) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()

def calculate_perplexity(loss):
    """Calculate perplexity from cross entropy loss"""
    return math.exp(loss.item())

def calculate_precision_recall_f1(output, target, ignore_index=0):
    """Calculate precision, recall, and F1 score"""
    mask = target != ignore_index
    predictions = output.argmax(dim=-1)
    
    # Flatten and apply mask
    predictions_flat = predictions[mask].view(-1)
    target_flat = target[mask].view(-1)
    
    # For multi-class classification, we'll calculate micro averages
    true_positives = (predictions_flat == target_flat).sum().item()
    total_predictions = predictions_flat.numel()
    total_actual = target_flat.numel()
    
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_actual if total_actual > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def generate_predictions(model, src_data, tgt_data, max_length=100):
    """Generate predictions using greedy decoding"""
    model.eval()
    with torch.no_grad():
        # Start with SOS token (assuming 1 is start token, 0 is pad)
        batch_size = src_data.size(0)
        generated = torch.ones(batch_size, 1, dtype=torch.long, device=src_data.device)  # SOS token
        
        for _ in range(max_length - 1):
            output = model(src_data, generated)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences generated EOS token (assuming 2 is EOS)
            if (next_token == 2).all():
                break
                
    return generated

# Add this evaluation function after your training loop
def comprehensive_evaluation(model, criterion, src_data, tgt_data, vocab_size):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    
    with torch.no_grad():
        # Get model predictions
        output = model(src_data, tgt_data[:, :-1])
        loss = criterion(output.contiguous().view(-1, vocab_size), 
                        tgt_data[:, 1:].contiguous().view(-1))
        
        # Calculate metrics
        accuracy = calculate_accuracy(output, tgt_data[:, 1:])
        perplexity = calculate_perplexity(loss)
        precision, recall, f1 = calculate_precision_recall_f1(output, tgt_data[:, 1:])
        
        # Generate sequences for BLEU score calculation
        generated_sequences = generate_predictions(model, src_data, tgt_data)
        
        # Token-level statistics
        predictions = output.argmax(dim=-1)
        unique_tokens_pred = len(torch.unique(predictions))
        unique_tokens_target = len(torch.unique(tgt_data[:, 1:]))
        
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'perplexity': perplexity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'unique_tokens_pred': unique_tokens_pred,
        'unique_tokens_target': unique_tokens_target
    }



# Add this visualization function
def plot_training_metrics(metrics_history):
    """Plot training metrics over time"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss and perplexity
    epochs = range(1, len(metrics_history) + 1)
    losses = [m['loss'] for m in metrics_history]
    perplexities = [m['perplexity'] for m in metrics_history]
    
    ax1.plot(epochs, losses, 'b-', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title('Training Loss')
    
    ax2.plot(epochs, perplexities, 'r-', label='Perplexity')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_title('Perplexity')
    
    # Plot accuracy and BLEU
    accuracies = [m['accuracy'] for m in metrics_history]
    
    ax3.plot(epochs, accuracies, 'g-', label='Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.set_title('Accuracy')
    
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Modify your training loop to track metrics
metrics_history = []

for epoch in range(100):
    transformer.train()
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), 
                    tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    
    # Calculate metrics every 10 epochs
    if (epoch + 1) % 10 == 0:
        transformer.eval()
        metrics = comprehensive_evaluation(transformer, criterion, src_data, tgt_data, tgt_vocab_size)
        metrics_history.append(metrics)
        
        print(f"\nEpoch: {epoch+1}")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        print(f"Unique Tokens - Pred: {metrics['unique_tokens_pred']}, Target: {metrics['unique_tokens_target']}")

# Final comprehensive evaluation
print("\n=== FINAL EVALUATION ===")
final_metrics = comprehensive_evaluation(transformer, criterion, val_src_data, val_tgt_data, tgt_vocab_size)

for metric, value in final_metrics.items():
    if isinstance(value, float):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    else:
        print(f"{metric.replace('_', ' ').title()}: {value}")

#
# Example of generating sample translations
def show_sample_translations(model, src_data, tgt_data, num_samples=3):
    """Show sample translations"""
    model.eval()
    with torch.no_grad():
        generated = generate_predictions(model, src_data, tgt_data)
        
        print("\n=== SAMPLE TRANSLATIONS ===")
        for i in range(min(num_samples, src_data.size(0))):
            print(f"Source: {src_data[i][:10].tolist()}...")  # First 10 tokens
            print(f"Target: {tgt_data[i][:10].tolist()}...")
            print(f"Generated: {generated[i][:10].tolist()}...")
            print("-" * 50)

# Show sample translations
show_sample_translations(transformer, val_src_data, val_tgt_data)

# Save model with metrics
checkpoint = {
    'model_state_dict': transformer.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_metrics': final_metrics,
    'metrics_history': metrics_history
}

print("Model saved with evaluation metrics!")

