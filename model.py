

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchaudio import transforms
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
import torchaudio


class EncoderBlock(nn.Module):
    """
    An encoder block for a Transformer-based model.

    The encoder block consists of a multi-head attention layer followed by a feedforward neural network.
    Layer normalization and residual connections are used to improve the stability and performance of the model.

    Arguments:
    ---------
        embedding_size : int
                       The size of the input and output embeddings.
        dim_feedforward : int
                        The size of the hidden layer in the feedforward neural network.

    """
    def __init__(self,embedding_size,dim_feedforward):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.attn = torch.nn.MultiheadAttention(embed_dim=embedding_size,num_heads=4,dropout=0.1,batch_first=True)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_size)
        self.linear1 = nn.Linear(embedding_size,dim_feedforward)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward,embedding_size)
        self.dropout3 = torch.nn.Dropout(0.1)

    def forward(self,x,attention_mask=None,key_padding_mask=None):
        """
        Forward pass through the encoder block.

        Arguments:
        ---------
            x : torch.Tensor
              The input tensor of shape (batch_size, sequence_length, embedding_size).
            attention_mask : torch.Tensor , optional
                      The attention mask tensor, with shape (sequence_length, sequence_length).
                      Defaults to None.
            key_padding_mask : torch.Tensor, optional
                             The key padding mask tensor, with shape (batch_size, sequence_length).
                             Defaults to None.
        """
        residual = x
        x_t = self.norm1(x)
        x_t, _ = self.attn(query=x_t,key=x_t,value=x_t,attn_mask=attention_mask,key_padding_mask=key_padding_mask)
        x_t = self.dropout1(x_t)
        x = residual + x_t
        residual = x
        x_t = self.norm2(x)
        x_t = self.linear1(x_t)
        x_t = F.relu(x_t)
        x_t = self.dropout2(x_t)
        x_t = self.linear2(x_t)
        x_t = self.dropout3(x_t)
        x = residual + x_t
        return x


class DecoderBlock(nn.Module):
    """
    A decoder block for a Transformer-based model.

    The decoder block consists of a self-attention layer, a cross-attention layer, and a feedforward neural network.
    Layer normalization and residual connections are used to improve the stability and performance of the model.

    Arguments:
    ---------
        embedding_size : int
                       The size of the input and output embeddings.
        dim_feedforward : int
                        The size of the hidden layer in the feedforward neural network.
    """
    def __init__(self,embedding_size,dim_feedforward):
        super(DecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embedding_size)
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=embedding_size,num_heads=4,dropout=0.1,batch_first=True)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(normalized_shape=embedding_size)
        self.attn = torch.nn.MultiheadAttention(embed_dim=embedding_size,num_heads=4,dropout=0.1,batch_first=True)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(normalized_shape=embedding_size)
        self.linear1 = nn.Linear(embedding_size,dim_feedforward)
        self.dropout3 = torch.nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward,embedding_size)
        self.dropout4 = torch.nn.Dropout(0.1)

    def forward(self,x,memory,x_attention_mask=None,x_key_padding_mask=None,memory_attention_mask=None,memory_key_padding_mask=None):
        """
        Forward pass through the decoder block.

        Arguments:
        ---------
            x : torch.Tensor
              The input tensor of shape (batch_size, sequence_length, embedding_size).
            memory : torch.Tensor
                   The memory tensor from the encoder, of shape (batch_size, memory_length, embedding_size).
            x_attn_mask : torch.Tensor, optional
                        The attention mask tensor for the input tensor `x`,
                        with shape (sequence_length, sequence_length). Defaults to None.
            x_key_padding_mask : torch.Tensor, optional
                               The key padding mask tensor for the input tensor `x`,
                               with shape (batch_size, sequence_length). Defaults to None.
            memory_attn_mask : torch.Tensor, optional
                             The attention mask tensor for the memory tensor,
                             with shape (sequence_length, memory_length). Defaults to None.
            memory_key_padding_mask : torch.Tensor, optional
                                    The key padding mask tensor for the memory tensor,
                                    with shape (batch_size, memory_length). Defaults to None.
        """
        x_t, _ = self.self_attn(query=x,key=x,value=x,attn_mask=x_attention_mask,key_padding_mask=x_key_padding_mask)
        x_t = self.dropout1(x_t)
        x = self.norm1(x + x_t)
        x_t, _ = self.attn(query=x,key=memory,value=memory,attn_mask=memory_attention_mask,key_padding_mask=memory_key_padding_mask)
        x_t = self.dropout2(x_t)
        x = self.norm2(x + x_t)
        x_t = self.linear1(x)
        x_t = F.relu(x_t)
        x_t = self.dropout3(x_t)
        x_t = self.linear2(x_t)
        x_t = self.dropout4(x_t)
        x = self.norm3(x + x_t)
        return x


class EncoderPreNet(nn.Module):
    """
    The purpose of the EncoderPreNet class is to preprocess the input text before it is passed to the encoder part of the Transformer-based TTS
    model. The preprocessing involves embedding the text, applying convolutional layers, and finally transforming the embeddings to the desired
    size.

    Arguments:
    ----------
    embedding_size : The size of the input and output embeddings.
    encoder_embedding_size : This parameter represents the size of the intermediate feature representations
                            (embeddings) that the EncoderPreNet will learn. It determines the number of output
                            channels for the convolutional layers.
    encoder_kernel_size : The size of the convolutional kernels used in the EncoderPreNet.
                         It determines the receptive field of the convolutional layers, i.e., the amount of input
                         context that each output feature will have access to.

    """
    def __init__(self,text_num_embeddings,encoder_embedding_size,embedding_size,encoder_kernel_size):
        super(EncoderPreNet, self).__init__()

        self.embedding = nn.Embedding( num_embeddings=text_num_embeddings,embedding_dim=encoder_embedding_size)
        self.linear1 = nn.Linear(encoder_embedding_size,encoder_embedding_size)
        self.linear2 = nn.Linear(encoder_embedding_size,embedding_size)
        self.conv1 = nn.Conv1d(encoder_embedding_size,encoder_embedding_size,kernel_size=encoder_kernel_size,stride=1,padding=int((encoder_kernel_size - 1) / 2),dilation=1)
        self.bn1 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(encoder_embedding_size,encoder_embedding_size,kernel_size=encoder_kernel_size,stride=1,padding=int((encoder_kernel_size - 1) / 2),dilation=1)
        self.bn2 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(encoder_embedding_size,encoder_embedding_size,kernel_size=encoder_kernel_size,stride=1,padding=int((encoder_kernel_size - 1) / 2),dilation=1)
        self.bn3 = nn.BatchNorm1d(encoder_embedding_size)
        self.dropout3 = torch.nn.Dropout(0.5)

    def forward(self, text):
        """
        Arguments:
        ---------
        text: torch.Tensor
            The tensor of token IDs, where each row represents a sample in the batch, and each column represents a token in the input sequence.
        """
        x = self.embedding(text)
        x = self.linear1(x)

        x = x.transpose(2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = x.transpose(1, 2)
        x = self.linear2(x)

        return x


class PostNet(nn.Module):
    """
    The purpose of the PostNet class is to further refine the mel spectrogram output from the previous module in the TTS model.
    It uses a series of 1D convolutional layers, batch normalization, and dropout to apply non-linear transformations to the input,
    effectively post-processing the mel spectrogram to improve its quality.

    Arguments:
    ---------
    postnet_embedding_size: int
        This parameter represents the number of output channels (or the embedding size) for the convolutional
        layers in the PostNet. It determines the size of the intermediate feature representations that the
        PostNet will learn during the forward pass.
    postnet_kernel_size: int
        This parameter represents the size of the convolutional kernels used in the PostNet
    mel_freq: int
        This parameter represents the number of mel frequency bins in the input mel spectrogram.

    """
    def __init__(self,mel_freq,postnet_embedding_size,postnet_kernel_size):
        super(PostNet, self).__init__()

        self.conv1 = nn.Conv1d(mel_freq,postnet_embedding_size,kernel_size=postnet_kernel_size,stride=1,padding=int((postnet_kernel_size - 1) / 2),dilation=1)
        self.bn1 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(postnet_embedding_size,postnet_embedding_size,kernel_size=postnet_kernel_size,stride=1,padding=int((postnet_kernel_size - 1) / 2),dilation=1)
        self.bn2 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.conv3 = nn.Conv1d(postnet_embedding_size,postnet_embedding_size,kernel_size=postnet_kernel_size,stride=1,padding=int((postnet_kernel_size - 1) / 2),dilation=1)
        self.bn3 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.conv4 = nn.Conv1d(postnet_embedding_size,postnet_embedding_size,kernel_size=postnet_kernel_size,stride=1,padding=int((postnet_kernel_size - 1) / 2),dilation=1)
        self.bn4 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.conv5 = nn.Conv1d(postnet_embedding_size,postnet_embedding_size,kernel_size=postnet_kernel_size,stride=1,padding=int((postnet_kernel_size - 1) / 2),dilation=1)
        self.bn5 = nn.BatchNorm1d(postnet_embedding_size)
        self.dropout5 = torch.nn.Dropout(0.5)
        self.conv6 = nn.Conv1d(postnet_embedding_size,mel_freq,kernel_size=postnet_kernel_size,stride=1,padding=int((postnet_kernel_size - 1) / 2),dilation=1)
        self.bn6 = nn.BatchNorm1d(mel_freq)
        self.dropout_6 = torch.nn.Dropout(0.5)

    def forward(self, x):
        """
        x : torch.Tensor
          The input x is first transposed to have the shape (N, FREQ, TIME), where N is the batch size,
          FREQ is the number of mel frequency bins, and TIME is the number of time steps. The input is then
          passed through the six sets of convolutional, batch normalization, and dropout layers. Each layer
          applies a non-linear transformation to the input, with the final layer outputting a tensor of
          size (N, FREQ, TIME). The output tensor is then transposed back to the original shape (N, TIME, FREQ)
          and returned
        """
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.tanh(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.tanh(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.tanh(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = torch.tanh(x)
        x = self.dropout5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.dropout_6(x)
        x = x.transpose(1, 2)
        return x


class DecoderPreNet(nn.Module):
    """
    The purpose of the DecoderPreNet is to apply a transformation to the target
    mel spectrogram before it is fed into the Decoder. This preprocessing can help
    the model learn a better representation of the input, which can improve the overall
    performance of the TTS system.

    Arguments:
    ---------
    mel_freq: int
        The mel_freq parameter represents the number of mel frequency bins in the mel spectrogram.
        This determines the resolution of the frequency representation.
    embedding_size: int
        The embedding_size parameter represents the dimensionality of the
        learned embeddings or feature representations in the model.

    """
    def __init__(self,mel_freq,embedding_size):
        super(DecoderPreNet, self).__init__()
        self.linear1 = nn.Linear(mel_freq,embedding_size)
        self.linear2 = nn.Linear(embedding_size,embedding_size)

    def forward(self, x):
        """
        Arguments:
        ---------
        x: torch.Tensor
            The target mel spectrogram
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)
        x = self.linear2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=True)
        return x

class TransformerTTS(nn.Module):
    """
    A Transformer-based Text-to-Speech (TTS) model.

    The TransformerTTS model is a Transformer-based architecture for generating mel spectrograms from input text.
    It consists of an encoder that processes the input text, a decoder that generates the output mel spectrogram,
    and a postnet that refines the generated mel spectrogram.

    Arguments:
    ---------
        device : str, optional
                The device to run the model on.
        max_mel_time : int, optional
                     The maximum length of the output mel spectrogram. Defaults to 1024.
        embedding_size : int, optional
                       The size of the input and output embeddings. Defaults to 256.
        mel_freq : int, optional
                 The number of mel frequency bins. Defaults to 128.
        dim_feedforward : int, optional
                        The size of the hidden layer in the feedforward neural networks. Defaults to 1024.
        encoder_embedding_size : int, optional
                               The size of the encoder's input embeddings. Defaults to 512.
        encoder_kernel_size : int, optional
                            The kernel size of the encoder's convolutional layers. Defaults to 3.
        postnet_embedding_size : int, optional
                               The size of the postnet's hidden layer. Defaults to 1024.
        postnet_kernel_size : int, optional
                            The kernel size of the postnet's convolutional layers. Defaults to 5.
    """
    def __init__(self, device,text_num_embeddings,embedding_size,encoder_embedding_size,dim_feedforward,postnet_embedding_size,encoder_kernel_size,postnet_kernel_size,max_mel_time,mel_freq):
        super(TransformerTTS, self).__init__()
        self.device = device
        self.text_num_embeddings = text_num_embeddings
        self.embedding_size = embedding_size
        self.encoder_embedding_size = encoder_embedding_size
        self.dim_feedforward = dim_feedforward
        self.postnet_embedding_size = postnet_embedding_size
        self.encoder_kernel_size = encoder_kernel_size
        self.postnet_kernel_size = postnet_kernel_size
        self.max_mel_time = max_mel_time
        self.mel_freq=mel_freq
        self.encoder_prenet = EncoderPreNet(self.text_num_embeddings,self.encoder_embedding_size,self.embedding_size,self.encoder_kernel_size)
        self.decoder_prenet = DecoderPreNet(self.mel_freq,self.embedding_size)
        self.postnet = PostNet(self.mel_freq,self.postnet_embedding_size,self.postnet_kernel_size)
        self.pos_encoding = nn.Embedding(num_embeddings=self.max_mel_time,embedding_dim=self.embedding_size)
        self.eb_1 = EncoderBlock(self.embedding_size,self.dim_feedforward)
        self.eb_2 = EncoderBlock(self.embedding_size,self.dim_feedforward)
        self.eb_3 = EncoderBlock(self.embedding_size,self.dim_feedforward)
        self.db_1 = DecoderBlock(self.embedding_size,self.dim_feedforward)
        self.db_2 = DecoderBlock(self.embedding_size,self.dim_feedforward)
        self.db_3 = DecoderBlock(self.embedding_size,self.dim_feedforward)
        self.linear1 = nn.Linear(self.embedding_size, self.mel_freq)
        self.linear2 = nn.Linear(self.embedding_size, 1)
        self.norm_memory = nn.LayerNorm(normalized_shape=self.embedding_size)

    def forward(self,text,text_len,mel,mel_len):
        """
        Forward pass through the TransformerTTS model.

        Arguments:
        ---------
            text : torch.Tensor
                 The input text tensor of shape (batch_size, sequence_length).
            text_len : torch.Tensor
                     The lengths of the input text sequences, of shape (batch_size,).
            mel : torch.Tensor
                The target mel spectrogram tensor of shape (batch_size, time_steps, mel_freq).
            mel_len : torch.Tensor
                    The lengths of the target mel spectrograms, of shape (batch_size,).
        """
        N = text.shape[0]
        S = text.shape[1]
        TIME = mel.shape[1]
        self.src_key_padding_mask = torch.zeros((N, S),device=text.device).masked_fill(~maskFromSequenceLengths(text_len,max_length=S),float("-inf"))
        self.src_mask = torch.zeros((S, S),device=text.device).masked_fill(torch.triu(torch.full((S, S),True,dtype=torch.bool),diagonal=1).to(text.device),float("-inf"))
        self.tgt_key_padding_mask = torch.zeros((N, TIME),device=mel.device).masked_fill(~maskFromSequenceLengths(mel_len,max_length=TIME),float("-inf"))
        self.tgt_mask = torch.zeros((TIME, TIME),device=mel.device).masked_fill(torch.triu(torch.full((TIME, TIME),True,device=mel.device,dtype=torch.bool),diagonal=1),float("-inf"))
        self.memory_mask = torch.zeros((TIME, S),device=mel.device).masked_fill(torch.triu(torch.full((TIME, S),True,device=mel.device,dtype=torch.bool),diagonal=1),float("-inf"))
        text_x = self.encoder_prenet(text)
        pos_codes = self.pos_encoding(torch.arange(self.max_mel_time).to(mel.device))
        S = text_x.shape[1]
        text_x = text_x + pos_codes[:S]
        text_x = self.eb_1(text_x,attention_mask=self.src_mask,key_padding_mask=self.src_key_padding_mask)
        text_x = self.eb_2(text_x,attention_mask=self.src_mask,key_padding_mask=self.src_key_padding_mask)
        text_x = self.eb_3(text_x,attention_mask=self.src_mask,key_padding_mask=self.src_key_padding_mask)
        text_x = self.norm_memory(text_x)
        mel_x = self.decoder_prenet(mel)
        mel_x = mel_x + pos_codes[:TIME]
        mel_x = self.db_1(x=mel_x,memory=text_x,x_attention_mask=self.tgt_mask,x_key_padding_mask=self.tgt_key_padding_mask,memory_attention_mask=self.memory_mask,memory_key_padding_mask=self.src_key_padding_mask)
        mel_x = self.db_2(x=mel_x,memory=text_x,x_attention_mask=self.tgt_mask,x_key_padding_mask=self.tgt_key_padding_mask,memory_attention_mask=self.memory_mask,memory_key_padding_mask=self.src_key_padding_mask)
        mel_x = self.db_3(x=mel_x,memory=text_x,x_attention_mask=self.tgt_mask,x_key_padding_mask=self.tgt_key_padding_mask,memory_attention_mask=self.memory_mask,memory_key_padding_mask=self.src_key_padding_mask)  # (N, TIME, E)
        mel_linear = self.linear1(mel_x)
        mel_postnet = self.postnet(mel_linear)
        mel_postnet = mel_linear + mel_postnet
        stop_token = self.linear2(mel_x)
        bool_mel_mask = self.tgt_key_padding_mask.ne(0).unsqueeze(-1).repeat(1, 1, self.mel_freq)
        mel_linear = mel_linear.masked_fill(bool_mel_mask,0)
        mel_postnet = mel_postnet.masked_fill(bool_mel_mask,0)
        stop_token = stop_token.masked_fill(bool_mel_mask[:, :, 0].unsqueeze(-1),1e3).squeeze(2)
        return mel_postnet, mel_linear, stop_token

    @torch.no_grad()
    def inference(self, text, max_length=800, stop_token_threshold=1e-5, with_tqdm=True):
        self.eval()
        self.train(False)
        textLengths = torch.tensor(text.shape[1]).unsqueeze(0).to(self.device)
        N = 1
        SOS = torch.zeros((N, 1, self.mel_freq)).to(self.device)

        melPadded = SOS
        melLengths = torch.tensor(1).unsqueeze(0).to(self.device)
        stop_token_outputs = torch.FloatTensor([]).to(text.device)

        if with_tqdm:
            iters = tqdm(range(max_length))
        else:
            iters = range(max_length)

        for _ in iters:
            mel_postnet, mel_linear, stop_token = self(
                text.to(self.device),
                textLengths.to(self.device),
                melPadded.to(self.device),
                melLengths.to(self.device)
            )

            melPadded = torch.cat(
                [
                    melPadded.to(self.device),
                    mel_postnet[:, -1:, :].to(self.device)
                ],
                dim=1
            )
            if torch.sigmoid(stop_token[:, -1]) > stop_token_threshold:
                break
            else:
                stop_token_outputs = torch.cat([stop_token_outputs.to(self.device), stop_token[:,-1:].to(self.device)], dim=1)
                melLengths = torch.tensor(
                    melPadded.shape[1]).unsqueeze(0).to(self.device)

        return mel_postnet,stop_token_outputs





LossStats = namedtuple(
    "TTSLoss", "loss mel_loss stop_token_loss"
)


class TTSLoss(torch.nn.Module):
    """
    TTSLoss is a custom loss function designed for training text-to-speech (TTS) models.
    It combines multiple loss components including Mean Squared Error (MSE) loss and
    Binary Cross Entropy (BCE) loss to optimize the model's performance.

    Arguments:
    ---------
    None

    Methods:
    --------
    forward:
        Computes the loss for a batch of predictions and targets.

    """
    def __init__(self):
        super(TTSLoss, self).__init__()

        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        mel_postnet_out,
        mel_out,
        stop_token_out,
        mel_target,
        stop_token_target
    ):
        """
        Compute the loss for a batch of predictions and targets.

        Arguments:
        ----------
        mel_postnet_out: torch.Tensor
            Predicted mel-spectrogram after the post-processing network.

        mel_out: torch.Tensor
            Predicted mel-spectrogram.

        stop_token_out: torch.Tensor
            Predicted stop token indicating the end of the audio sequence.

        mel_target: torch.Tensor
            Target mel-spectrogram.

        stop_token_target: torch.Tensor
            Target stop token.

        Returns:
        --------
        LossStats:
            A named tuple containing the total loss, mel-spectrogram loss, and stop token loss.
        """
        stop_token_target = stop_token_target.view(-1, 1)

        stop_token_out = stop_token_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_postnet_out, mel_target)

        stop_token_loss = self.bce_loss(
            stop_token_out, stop_token_target) * 1.0

        return LossStats(mel_loss + stop_token_loss, mel_loss, stop_token_loss)



def maskFromSequenceLengths(
    seq_lengths: torch.Tensor,
    max_length: int
) -> torch.BoolTensor:
    """
        Create a mask tensor from the sequence lengths.

       Arguments:
       ---------
            seq_lengths : torch.Tensor
                        The lengths of the sequences, of shape (batch_size,).
            max_length : int
                       The maximum length of the sequences.
        example :
        --------
        our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
        `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.
    """
    ones = seq_lengths.new_ones(seq_lengths.size(0), max_length)
    rangeTensor = ones.cumsum(dim=1)
    return seq_lengths.unsqueeze(1) >= rangeTensor




def TTSCollate(batch):
    """Collates a batch of text and mel spectrogram data for text-to-speech tasks.

    Arguments:
    ---------
        batch: A list of dictionaries, each containing 'text' and 'mel' tensors.

    Returns:
    -------
        A tuple containing:
            - text: Original text tensor (for reference).
            - textsPadded: Padded text tensor.
            - textLengths: Tensor of text lengths.
            - melsPadded: Padded mel spectrogram tensor.
            - melLengths: Tensor of mel spectrogram lengths.
            - stopTokenPadded: Padded stop token indicating sequence end.
    """
    maximumTextLength = torch.tensor([item['text'].shape[-1] for item in batch],dtype=torch.int32).max()
    MaximumMelLength = torch.tensor([item['mel'].shape[-1] for item in batch],dtype=torch.int32).max()

    textLengths = []
    melLengths = []
    textsPadded = []
    melsPadded = []

    for item in batch:
        text = item['text']
        mel = item['mel']
        textLength = item["text"].shape[-1]
        textPadded = torch.nn.functional.pad(text,pad=[0, maximumTextLength-textLength],value=0)
        melLength = item["mel"].shape[-1]
        melPadded = torch.nn.functional.pad(mel,pad=[0, MaximumMelLength-melLength],value=0)
        textLengths.append(textLength)
        melLengths.append(melLength)
        textsPadded.append(textPadded)
        melsPadded.append(melPadded)

    textLengths = torch.tensor(textLengths, dtype=torch.int32)
    melLengths = torch.tensor(melLengths, dtype=torch.int32)
    textsPadded = torch.stack(textsPadded, 0)
    melsPadded = torch.stack(melsPadded, 0).transpose(1, 2)

    stopTokenPadded = maskFromSequenceLengths(
        melLengths,
        MaximumMelLength
    )
    stopTokenPadded = (~stopTokenPadded).float()
    stopTokenPadded[:, -1] = 1.0

    return text, \
        textsPadded, \
        textLengths, \
        melsPadded, \
        melLengths, \
        stopTokenPadded \


class AudioFeatureExtractor:
    """
    n_fft : int
          Size of the Fast Fourier Transform (FFT) window.
    win_length : int
               Length of the window function applied to each frame.
    hop_length : int
               Number of samples to hop between frames.
    power : float
          Power exponent used in spectrogram calculation (e.g., 2 for power spectrogram).
    n_mels : int
           Number of Mel filters.
    sample_rate : int
                Sampling rate of the audio signal.
    n_stft : int
           Optional number of STFT (Short-Time Fourier Transform) windows, defaults to None (computed from audio length and hop length).
    max_db : float
           Maximum decibel value for normalization.
    norm_db : bool
            Whether to normalize Mel spectrogram to decibels before scaling.
    ampl_multiplier : float
                    Multiplier for amplitude-to-dB conversion.
    ampl_amin : float
              Minimum amplitude value for log-scaling in dB conversion.
    db_multiplier : float
                  Multiplier for scaling decibel values.
    scale_db : float
             Scaling factor for Mel spectrogram in decibels.
    ampl_ref : float
             Reference amplitude for dB-to-amplitude conversion.
    ampl_power : float
               Power for dB-to-amplitude conversion.
    mel_freq : int
             Optional number of Mel bins, defaults to n_mels (derived from Mel filterbank).
    """
    def __init__(self,sr = 22050,n_fft = 2048,n_stft = 1025,frame_shift = 0.0125,
                 hop_length = 256,frame_length = 0.05,win_length = 1024,mel_freq = 128,
                 max_mel_time = 1024,max_db = 100,scale_db = 10,ref = 4.0,power = 2.0,norm_db = 10 ,
                 ampl_multiplier = 10.0,ampl_amin = 1e-10,db_multiplier = 1.0,ampl_ref = 1.0,
                 ampl_power = 1.0):

        self.sr=sr
        self.n_fft=n_fft
        self.n_stft=n_stft
        self.frame_shift=frame_shift
        self.hop_length=hop_length
        self.frame_length=frame_length
        self.win_length=win_length
        self.mel_freq=mel_freq
        self.max_mel_time=max_mel_time
        self.max_db=max_db
        self.scale_db=scale_db
        self.ref=ref
        self.power=power
        self.norm_db=norm_db
        self.ampl_multiplier=ampl_multiplier
        self.ampl_amin=ampl_amin
        self.db_multiplier=db_multiplier
        self.ampl_ref=ampl_ref
        self.ampl_power=ampl_power

        self.symbols = [
                'EOS', ' ', '!', ',', '-', '.', \
                ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', \
                'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', \
                'â', 'è', 'é', 'ê', 'ü', '’', '“', '”' \
            ]

        self.specTransform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=power
        )

        self.melScaleTransform = torchaudio.transforms.MelScale(
        n_mels=mel_freq,
        sample_rate=sr,
        n_stft=n_stft
        )

        self.InverseMelTransform = torchaudio.transforms.InverseMelScale(
        n_mels=mel_freq,
        sample_rate=sr,
        n_stft=n_stft
        )

        self.griffnlimTransform = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length
        )

        self.symbol_to_id = {
            s: i for i, s in enumerate(self.symbols)
        }


    def text_to_seq(self,text):
        text = text.lower()
        seq = []
        for s in text:
            _id = self.symbol_to_id.get(s, None)
            if _id is not None:
                seq.append(_id)

        seq.append(self.symbol_to_id["EOS"])
        return torch.IntTensor(seq)


    def powerToDbMelSpec(self,mel_spec):
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec,
            multiplier = self.ampl_multiplier,
            amin = self.ampl_amin,
            db_multiplier = self.db_multiplier,
            top_db = self.max_db
        )
        mel_spec = mel_spec/self.scale_db
        return mel_spec


    def dbToPowerMelSpec(self,mel_spec):
        mel_spec = mel_spec*self.scale_db
        mel_spec = torchaudio.functional.DB_to_amplitude(
            mel_spec,
            ref=self.ampl_ref,
            power=self.ampl_power
        )
        return mel_spec


    def GenerateToMelSpectogram(self,wav):
        spec = self.specTransform(wav)
        mel_spec = self.melScaleTransform(spec)
        db_mel_spec = self.powerToDbMelSpec(mel_spec)
        db_mel_spec = db_mel_spec.squeeze(0)
        return db_mel_spec


    def inverse_mel_spec_to_wav(self,mel_spec):
        power_mel_spec = self.dbToPowerMelSpec(mel_spec)
        spectrogram = self.InverseMelTransform(power_mel_spec)
        generated_wav = self.griffnlimTransform(spectrogram)
        return generated_wav
