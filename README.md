# TTS TRANSFORMER


### How to run
* just download the dataset using the .ipynb file
* download the trained model using following link and place the result folder in the folder which containes .ipynb file and dataset
* link : https://drive.google.com/file/d/1AI-h-0PnFXmLc8J1A1Y23buUgCWJ-L9o/view?usp=drive_link
* using ipynb you can test everything 
##Introduction
Despite breakthroughs in text-to-speech (TTS) synthesis, producing natural-sounding speech remains difficult. Traditional approaches fail to capture long-range relationships in text while keeping natural prosody, limiting their usefulness in practical applications. High-quality TTS systems have several applications, including assisting visually impaired users, creating audiobooks, and improving voice interfaces. To reach their full potential, TTS systems must use natural speech patterns and be computationally efficient. The main issues in TTS synthesis include capturing long-range relationships in text, balancing speech accuracy and naturalness, and assuring real-time creation for practical use cases. Previous TTS techniques, such Recurrent Neural Networks (RNNs) and Tacotron2, have made great progress. However, RNNs suffer with long-range dependencies, and Tacotron2, despite its advances, remains computationally costly. This study presents a transformer-based TTS model for the SpeechBrain framework. Transformers, noted for their capacity to capture long-range dependencies, appear to be a possible answer to the issues of TTS synthesis. Building on Tacotron2's design, use of SpeechBrain's efficiency and efficacy while implementing the model. Transformers have shown excellent performance in a variety of sequence-to-sequence tasks, making them ideal candidates for TTS synthesis. SpeechBrain offers a strong infrastructure for developing and optimizing TTS models, guaranteeing rapid development and deployment. The goal is to train and test the framework on the LJSpeech dataset, a popular benchmark for TTS synthesis. Performance evaluation will include comparisons to Tacotron2 based on human evaluation measures such as naturalness and audio quality. This project requires access to GPUs for efficient training, used Colab Pro v100 as computational resources.

### MODEL

Implementation of Transformer TTS model
<img width="288" alt="image" src="https://github.com/PRIYANG012/TTS_Transformer/assets/45707603/81c56c7a-9981-4829-a805-7c969b11a2ae">


(IMPORTANT) STOP TOKEN : The stop token mechanism is vital in autoregressive models, particularly in text-to-speech systems. It functions as a signal to indicate when audio creation should stop. The stop token tensor is organised similarly to an aligned spectrogram, and its length corresponds to the time dimension.


