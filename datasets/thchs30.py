from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import glob
#from util import audio
from datasets import audio
from hparams import hparams as hp
from wavenet_vocoder.util import is_mulaw, is_mulaw_quantize, mulaw, mulaw_quantize


#def build_from_path(hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
def build_from_path(hparams, input_dirs, out_dir, n_jobs=1, tqdm=lambda x: x):
  '''Preprocesses the THCHS30 dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the THCHS30 dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=n_jobs)
  futures = []
  index = 1

  for input_dir in input_dirs:
    trn_files = glob.glob(os.path.join(input_dir, 'data', '*.trn'))

    for trn in trn_files:
      with open(trn) as f:
        f.readline()
        pinyin = f.readline().strip('\n')
        wav_file = trn[:-4]
        task = partial(_process_utterance, out_dir, index, wav_file, pinyin,hparams)
        futures.append(executor.submit(task))
        index += 1
    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(out_dir, index, wav_path, pinyin,hparams):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    wav_path: Path to the audio file containing the speech input
    pinyin: The pinyin of Chinese spoken in the input audio file

  Returns:
    A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
  '''
  
  mel_dir=out_dir + "/mels"
  linear_dir=out_dir + "/linear"
  wav_dir=out_dir + "/audio"
  
  # Load the audio to a numpy array:
  wav = audio.load_wav(wav_path,sr=hparams.sample_rate)
  print("debug wav_path:",wav_path)
  #rescale wav
  if hparams.rescale:
    wav = wav / np.abs(wav).max() * hparams.rescaling_max

  #M-AILABS extra silence specific
  if hparams.trim_silence:
    wav = audio.trim_silence(wav, hparams)

  #Mu-law quantize
  if is_mulaw_quantize(hparams.input_type):
    #[0, quantize_channels)
    out = mulaw_quantize(wav, hparams.quantize_channels)

    #Trim silences
    start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
    wav = wav[start: end]
    out = out[start: end]

    constant_values = mulaw_quantize(0, hparams.quantize_channels)
    out_dtype = np.int16

  elif is_mulaw(hparams.input_type):
    #[-1, 1]
    out = mulaw(wav, hparams.quantize_channels)
    constant_values = mulaw(0., hparams.quantize_channels)
    out_dtype = np.float32

  else:
    #[-1, 1]
    out = wav
    constant_values = 0.
    out_dtype = np.float32

  # Compute a mel-scale spectrogram from the wav:
  #mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
  mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
  mel_frames = mel_spectrogram.shape[1]
  if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
    print("debug --- drop wav_path:",wav_path, "mel_frames:",mel_frames)
    return None

  # Compute the linear-scale spectrogram from the wav:
  #spectrogram = audio.spectrogram(wav).astype(np.float32)
  #n_frames = spectrogram.shape[1]
  linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
  linear_frames = linear_spectrogram.shape[1]

  #sanity check
  assert linear_frames == mel_frames

  if hparams.use_lws:
    #Ensure time resolution adjustement between audio and mel-spectrogram
    fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
    l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

    #Zero pad audio signal
    out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
  else:
    #Ensure time resolution adjustement between audio and mel-spectrogram
    pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams))

    #Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, pad, mode='reflect')

  assert len(out) >= mel_frames * audio.get_hop_size(hparams)

  #time resolution adjustement
  #ensure length of raw audio is multiple of hop size so that we can use
  #transposed convolution to upsample
  out = out[:mel_frames * audio.get_hop_size(hparams)]
  assert len(out) % audio.get_hop_size(hparams) == 0
  time_steps = len(out)



  # Write the spectrograms to disk:
  #spectrogram_filename = 'thchs30-spec-%05d.npy' % index
  #mel_filename = 'thchs30-mel-%05d.npy' % index
  #np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
  #np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

  audio_filename = 'audio-{}.npy'.format(index)
  mel_filename = 'mel-{}.npy'.format(index)
  linear_filename = 'linear-{}.npy'.format(index)
  np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
  np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)
  print("debug save wav file:",os.path.join(wav_dir, audio_filename))
  # Return a tuple describing this training example:
  return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, pinyin)
