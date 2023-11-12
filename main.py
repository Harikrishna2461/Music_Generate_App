#This file can be run on replit,local or any other IDEs,but make sure Meta's 
#'audiocraft' library folder containing all the models and other required files is in 
#the same directory as the main.py file.
import base64
import os
import numpy as np
import streamlit as st
import torch
import torchaudio
from audiocraft.models import MusicGen

#Loading the Pretrained MusicGen Model from the audiocraft library of Meta,which takes 
#text inputs and Generate a Music File/Media according to the user text input.
#We are also caching the model so we dont need to load the model again and again during 
#runtime and reduce costs
def Gen_AI_Model():
  Gen_AI_Model = MusicGen.get_pretrained("facebook/musicgen-small")
  return Gen_AI_Model

#to generate the numerical tensors of the generate music audio.
def generate_music_tensors(description,duration:int):
  model = Gen_AI_Model()
  #Setting the Hyperparameters,refer to the MusicGen Model's Github page to see what 
  #exactly these parameters signify.Here 'use_sampling' is used to control whether the 
  #model samples from existing music or creates completely new music,top_k refers to 
  #top k number of musics which match the text description to sample from and duration
  #refers to the duration of the music generated.
  
  model.set_generation_params(use_sampling=False,top_k=250,duration=duration)
  generated_music = model.generate(descriptions=[description],progress=True,return_tokens=True)
  return generated_music[0]

#To load the generated_music_tensors into a music we use the following function.
#This Renders an audio player for the given audio samples and saves them to a local 
#directory.
#
"""
Arguements:
    samples (torch.Tensor): a Tensor of decoded audio samples
        with shapes [B, C, T] or [C, T]
    sample_rate (int): sample rate audio should be displayed with.
    save_path (str): path to the directory where audio should be saved.
"""
def load_music(samples: torch.Tensor):
  print("Samples (inside function): ", samples)
  sample_rate = 32000
  save_path = "audio_output/"
  assert samples.dim() == 2 or samples.dim() == 3

  samples = samples.detach().cpu()
  if samples.dim() == 2:
      samples = samples[None, ...]

  for idx, audio in enumerate(samples):
      audio_path = os.path.join(save_path, f"audio_{idx}.wav")
      torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

#Using a github emoji for music as the page icon.
st.set_page_config(page_title="Music Generator",page_icon=":musical_note:")

#Instructing the user what to do
st.write("Give text input to get the desired Music you want")

text_input = st.text_area("Give your Input")
#Setting the time slider to generate music for that many seconds as specified by the 
#user with minimum = 5 seconds,maximum = 360 seconds and default = 20 seconds
time_slider = st.slider("Give the duration of the music (in seconds)",5,20,360)

if text_input and time_slider : #When these fileds are not empty.
  st.json(
           {
           "description":text_input,
           "selected time duration (in seconds):":time_slider
          }
        )
  st.subheader("Generated Music")
  generated_music_tensor = generate_music_tensors(text_input, time_slider)
  print("Musci Tensors: ", generated_music_tensor)
  generated_music_file = load_music(generated_music_tensor)
  music_filepath = 'audio_output/audio_0.wav'
  audio_file = open(music_filepath, 'rb')
  audio_bytes = audio_file.read()
  st.audio(audio_bytes)
  st.markdown(get_binary_file(music_filepath, 'Audio'), unsafe_allow_html=True)

#The End.
  
   

