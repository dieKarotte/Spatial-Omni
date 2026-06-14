import torch
import torch.nn.functional as F
from torch import nn
from spatpy.signal_path.analysis import PowerVector, ForwardTransform, SignalPathConfig
from ufb_banding.banding.spatial import SpatialBandingCoefs, SpatialBandingParams
from ufb_banding.banding import BandingParams
from spatpy.signal_path.primitives import Reblocker
import matplotlib.pyplot as plt
import numpy as np
import math
import torchaudio
# from sscv import *
# from torchaudio.transforms import Gammatone, MelSpectrogram


def _stft_safe(audio: torch.Tensor, **kwargs):
    stft_dtype = audio.dtype
    if stft_dtype in (torch.float16, torch.bfloat16):
        stft_dtype = torch.float32
    window = kwargs.get("window")
    if window is not None and window.dtype != stft_dtype:
        kwargs["window"] = window.to(device=audio.device, dtype=stft_dtype)
    return torch.stft(audio.to(dtype=stft_dtype), **kwargs)


def cal_features(audio):
    # audio[:, 0, :]/torch.max(torch.abs(audio[:, 0, :]))
    ch1 = audio[:, :,0] # w channel
    # ch2 = audio[:, :,1] # x channel
    # STFT
    enc_ch1 = _stft_safe(ch1, n_fft=1536, hop_length=768, return_complex=True)
    
    # enc_ch2 = torch.stft(ch2, n_fft=1536, hop_length=768, return_complex=True)
    f = torch.view_as_real(enc_ch1)
    f = torch.sqrt(f[:, :, :, 0] ** 2 + f[:, :, :, 1] ** 2)  # Magnitude
    # Ipd ild calculation
    # cc = enc_ch1 * torch.conj(enc_ch2) # [1,769,84]
    #const=torch.ones(128,769,63)*1e-7
    #const=const.float().to(device="cuda")
    # ipd = cc /(torch.abs(cc)+10e-8)
    # ipd_ri = torch.view_as_real(ipd)
    # ild = torch.log(torch.abs(enc_ch1) + 10e-8) - torch.log(torch.abs(enc_ch2) + 10e-8)
    #x2 = torch.cat((ipd_ri[:, :, :, 0], ipd_ri[:, :, :, 1], ild), axis=1)

    return f

def mel_tri_filterbank(fs, Nbin, Nband):
    fmin = max(fs / Nbin, 300)
    fmax = min(fs / 2, 10000)

    fbin = torch.linspace(0, Nbin, Nbin) * fs / (Nbin * 2)

    mbin = 2595 * torch.log10(1 + fbin / 700)
    fmin_tensor = torch.tensor(fmin, dtype=torch.float32)
    fmax_tensor = torch.tensor(fmax, dtype=torch.float32)
    mband = torch.linspace(2595 * torch.log10(1 + fmin_tensor / 700), 
                           2595 * torch.log10(1 + fmax_tensor / 700), Nband)
    
    bandfreq = (10 ** (mband / 2595) - 1) * 700  # centre frequency of each band

    banding = torch.zeros((len(fbin), Nband))

    for b in range(len(fbin)):
        for band in range(Nband):
            banding[b, band] = max(1 - abs(mband[band] - mbin[b]) / (mband[1] - mband[0]), 0)
    
    return banding

def normalize_filters(filterbank):
    rms_per_filter = torch.sqrt(torch.mean(filterbank ** 2, dim=1))
    rms_normalization_values = 1. / (rms_per_filter / torch.max(rms_per_filter))
    normalized_filterbank = filterbank * rms_normalization_values[:, None]
    return normalized_filterbank

def gammatone_impulse_response(center_freq, t):
    erb = 24.7 + 0.108 * center_freq
    p = 2
    divisor = (math.pi * math.factorial(2 * p - 2) * 2 ** (-(2 * p - 2))) / (math.factorial(p - 1) ** 2)
    b = erb / divisor
    a = 1.0
    gammatone_ir = a * t ** (p - 1) * torch.exp(-2 * math.pi * b * t) * torch.cos(2 * math.pi * center_freq * t)
    gammatone_ir /= torch.sum(torch.abs(gammatone_ir))  # Normalize the filter gain
    return gammatone_ir

def create_gammatone_filterbank(frequencies, fs=16000, duration=0.0128):
    t = torch.linspace(1. / fs, duration, int(fs * duration))
    filterbank = []

    for center_freq in frequencies:
        impulse_response = gammatone_impulse_response(center_freq, t)
        filterbank.append(impulse_response)

    filterbank = torch.stack(filterbank)
    filterbank = normalize_filters(filterbank)
    return filterbank

def apply_filterbank_to_stft(spectrogram, filterbank):
    batch, freq_bins, frames = spectrogram.shape
    filter_num, filter_len = filterbank.shape
    # Expand dimensions for broadcasting
    spectrogram = spectrogram.reshape(batch*frames,1,freq_bins) # [batch, 1, freq_bins, frames]
    filterbank = filterbank.unsqueeze(1) # [filter_num, 1, filter_length]

    # Apply the filterbank to the STFT spectrogram
    filtered_spectrogram = F.conv1d(spectrogram, filterbank, padding=filter_len// 2, groups=1)
    # Sum across the frequency bins to get the banded output
    banded_spectrogram = torch.sum(filtered_spectrogram, dim=2)
    banded_spectrogram = banded_spectrogram.view(batch,filter_num,-1)
    return banded_spectrogram

def cal_pv_spec_feature(audio, pvmat, device):
    NChannels = audio.shape[1]
    pvmat = torch.tensor(pvmat).to(device).to(torch.complex64)
    # STFT 
    fs = 16000
    n_fft = int(0.08 * fs)       # FFT window size
    hop_length = int(0.04 * fs)  # Hop length (stride)
    win_length = int(0.08 * fs)  # Window size
    window = torch.hann_window(win_length).to(device)

    stfts = []
    for ch in range(NChannels):
        stft = _stft_safe(
            audio[:, ch, :],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        stfts.append(stft)
    
    stfts = torch.stack(stfts, dim=-1)  # [batch_size, time, freq, channels]

    # Define mel filter bank
    Nband = 33
    Nbin = stfts.shape[-3]
    banding = mel_tri_filterbank(fs, Nbin, Nband).to(device)

    T = stfts.shape[-2]
    batch_size = stfts.shape[0]
    
    w_banded_spectrogram = torch.zeros((batch_size, Nband, T), device=device)
    pv = torch.zeros((batch_size, NChannels**2, Nband, T), device=device)

    for batch in (range(batch_size)):
        for t in (range(T)):
            stft_slice = stfts[batch, :, t, :].T  # [freq, channels]
            sig = torch.einsum('tb,ct->tcb', torch.sqrt(banding), stft_slice)   # [batch_size, Nband, channels]
            cov = torch.einsum('tcb,tCb->cCb', sig, torch.conj(sig))  # [batch_size, NChannels, NChannels, Nband]
            pv0 = torch.real(pvmat @ cov.reshape(NChannels**2, -1))
            w_banded_spectrogram[batch, :, t] = torch.abs(torch.mean(sig[:, 0, :], dim=0))
            pv[batch,:, :, t] = pv0

    w_banded_spectrogram = w_banded_spectrogram.unsqueeze(1)
    return pv,w_banded_spectrogram

def transform_input(Y, epsilon=1e-8):
    return torch.log10(torch.abs(Y) + epsilon)

# only calulate the stft features for w channel only, return magnitude response and phase
def cal_w_spec_feature(audio, device):
    # STFT 
    fs = 16000
    # tried with 32ms window with 10ms overlapping
    n_fft = int(0.032 * fs)  
    hop_length = int(0.01 * fs)
    win_length = int(0.032 * fs)  
    window = torch.hann_window(win_length).to(device)

    w_audio = audio[:, :, 0]/torch.max(torch.abs(audio[:, :, 0]))
    w_stft = _stft_safe(
        w_audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    # banded_spectrogram = apply_filterbank_to_stft(torch.abs(w_stft), filterbank)
    # magnitude = banded_spectrogram.cpu().numpy()
    # plt.figure(figsize=(10, 6), dpi=300)  # High resolution
    # plt.imshow(20*np.log10(magnitude[0,:,:]), aspect='auto', origin='lower', cmap='inferno')
    # plt.colorbar(label='Magnitude (dB)')
    # plt.xlabel('Time Frame')
    # plt.ylabel('Frequency Bin')
    # plt.title('Magnitude of STFT')
    # plt.savefig('stft_magnitude.png', bbox_inches='tight')
    # plt.close()

    return torch.log10(torch.abs(w_stft)+1e-8)

def compute_mfcc_batch(batch_audio, sample_rate=16000, n_mfcc=20, frame_size=0.094, frame_stride=0.047, device='cuda'):
    # Define the MFCC transformation
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            'n_fft': int(sample_rate * frame_size),
            'hop_length': int(sample_rate * frame_stride),
            'n_mels': 40,
            'center': False
        }
    ).to(device)
    
    # Apply the transformation to each sample in the batch
    batch_mfcc = []
    for i in range(batch_audio.shape[0]):
        mfcc = mfcc_transform(batch_audio[i, :, 0])
        batch_mfcc.append(mfcc)
    
    # Stack MFCCs along the batch dimension
    batch_mfcc = torch.stack(batch_mfcc)
    return batch_mfcc

def gammatone_impulse_response(samplerate_hz, length_in_seconds, center_freq_hz, phase_shift):
    # Generate single parametrized gammatone filter
    p = 2  # filter order
    erb = 24.7 + 0.108 * center_freq_hz  # equivalent rectangular bandwidth
    divisor = (math.pi * math.factorial(2 * p - 2) * 2 ** (-(2 * p - 2))) / math.factorial(p - 1) ** 2
    b = erb / divisor  # bandwidth parameter
    a = 1.0  # amplitude. This is varied later by the normalization process.
    L = int(math.floor(samplerate_hz * length_in_seconds))
    t = torch.linspace(1. / samplerate_hz, length_in_seconds, L)
    gammatone_ir = a * t ** (p - 1) * torch.exp(-2 * math.pi * b * t) * torch.cos(
        2 * math.pi * center_freq_hz * t + phase_shift)
    return gammatone_ir

class GammatoneFilterbank(nn.Module):
    def __init__(self, num_filters, input_dim, output_dim, sample_rate, filter_length_in_seconds, phase_shift=0):
        super(GammatoneFilterbank, self).__init__()
        self.num_filters = num_filters
        self.sample_rate = sample_rate
        self.filter_length = int(math.floor(sample_rate * filter_length_in_seconds))  # Ensure length is an integer
        self.phase_shift = phase_shift
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filters = self.init_gammatone_filters(num_filters, sample_rate, filter_length_in_seconds, phase_shift)

        # Define a 1D convolution layer to reduce the dimension
        self.conv1d = nn.Conv1d(in_channels=num_filters, out_channels=output_dim, kernel_size=1)

    def init_gammatone_filters(self, num_filters, sample_rate, filter_length_in_seconds, phase_shift):
        min_freq = 50
        max_freq = 2000
        freqs = torch.logspace(math.log10(min_freq), math.log10(max_freq), num_filters)
        filters = []
        for f in freqs:
            filter_response = gammatone_impulse_response(sample_rate, filter_length_in_seconds, f, phase_shift)
            filters.append(filter_response)
        return torch.stack(filters)

    def forward(self, x):
        # Input shape: [batch_size, 1, input_dim]
        batch_size, num_channels, input_dim = x.shape

        # Ensure the filters are on the same device as the input tensor
        filters = self.filters.to(x.device)

        # Apply gammatone filters
        filtered_signals = []
        for filter in filters:
            filtered_signal = torch.nn.functional.conv1d(x, filter.view(1, 1, -1), padding=self.filter_length // 2)
            filtered_signals.append(filtered_signal)

        # Stack the filtered signals
        filtered_signals = torch.cat(filtered_signals, dim=1)  # [batch_size, num_filters, filtered_signal_length]

        # Reduce the dimension from num_filters to output_dim using a 1D convolution
        output = self.conv1d(filtered_signals)  # [batch_size, output_dim, filtered_signal_length]

        return output

def gammatone_filterbank(audio, sample_rate, num_bands, low_freq, high_freq, device):
    filters = []
    # filter_len = 0.2*sample_rate  # set the length of the gammatone filter to 0.2s
    audio_length = audio.shape[-1]
    for center_freq in torch.linspace(low_freq, high_freq, num_bands, device=device):
        t = torch.arange(audio_length, device=device) / sample_rate
        filter = torch.sin(2 * torch.pi * center_freq * t)
        filters.append(filter)
    filters = torch.stack(filters) 
    # audio = audio.unsqueeze(1)
    # # Reshape the filterbank to [20, 1, 3200] for convolution
    # filterbank = filters.unsqueeze(1)
    # # Apply 1D convolution with padding to keep the output length same as input length
    # output = F.conv1d(audio, filterbank, padding=filter_len // 2, groups=1)
    # gammatone_features = output[:, :, :audio_length]
    gammatone_features = torch.einsum('bf,cf->bcf', audio, filters)
    return gammatone_features # [batch,20,64000]

def compute_gammatone_features(audio, sample_rate=16000, num_bands=20, low_freq=50, high_freq=2000, frame_size=64, hop_size=32):
    # Convert audio to 1D if it is not
    device = audio.device
    audio = audio[:,:,0] # take the w channel only
    frame_count = (audio.shape[1] - frame_size) // hop_size + 1
    # Compute Gammatone features
    filter_length_in_seconds = 0.025
    gammatone_filterbank = GammatoneFilterbank(num_bands, frame_count, output_dim, sample_rate,
                                               filter_length_in_seconds)
    
    gammatone_features = gammatone_filterbank(audio, sample_rate, num_bands, low_freq, high_freq, device)
    
    # Compute the log-energy of frames
    frame_count = (gammatone_features.shape[-1] - frame_size) // hop_size + 1
    log_energy_frames = []

    for i in range(frame_count):
        frame = gammatone_features[:, :, i * hop_size:i * hop_size + frame_size]
        log_energy = torch.log(torch.mean(frame**2, dim=-1) + 1e-10)  # Adding epsilon to avoid log(0)
        log_energy_frames.append(log_energy)

    log_energy_frames = torch.stack(log_energy_frames, dim=-1) #[batch,20,1999]
    
   # Compute DFT (up to 500 Hz)
    n_fft = frame_size
    dft = _stft_safe(audio, n_fft=n_fft, hop_length=hop_size, return_complex=True)
    # freqs = torch.fft.fftfreq(n_fft, 1 / sample_rate)
    dft_500Hz = dft[:, :3,:]

    # Compute log-energy for DFT (up to 500 Hz)
    dft_500Hz_log_energy = torch.log(torch.mean(dft_500Hz.abs()**2,dim=1) + 1e-10).unsqueeze(1)

    # Compute magnitude-sorted DFT
    magnitude_sorted_dft, _ = torch.sort(dft.abs(), dim=1, descending=True)

    # Compute log-energy for magnitude-sorted DFT
    magnitude_sorted_dft_log_energy = torch.log(torch.mean(magnitude_sorted_dft**2,dim=1) + 1e-10).unsqueeze(1)

    # Compute Cepstrum
    cepstrum = torch.fft.ifft(torch.log1p(dft.abs())).abs()

    # Compute log-energy for Cepstrum
    cepstrum_log_energy = torch.log(torch.mean(cepstrum**2,dim=1) + 1e-10).unsqueeze(1)

    # Compute Envelope Follower
    envelope_follower = torch.abs(audio).unsqueeze(1)
    frame_size_env = min(frame_size, envelope_follower.size(-1))
    envelope_follower = F.conv1d(envelope_follower,
                                 torch.ones(1, 1, frame_size_env).to(device) / frame_size_env, stride=hop_size, padding=frame_size_env // 2)

    # Compute log-energy for Envelope Follower
    envelope_follower_log_energy = torch.log(torch.mean(envelope_follower**2,dim=1) + 1e-10).unsqueeze(1)
    
    # Compute Time-domain signal (low-passed)
    low_pass_filter = torch.tensor([1.0] * frame_size, device=device) / frame_size
    time_domain_low_passed = torch.nn.functional.conv1d(audio.unsqueeze(1), low_pass_filter.unsqueeze(0).unsqueeze(0), stride=hop_size)

    # Compute log-energy for Time-domain signal (low-passed)
    time_domain_low_passed_log_energy = torch.log(torch.mean(time_domain_low_passed**2,dim=1) + 1e-10).unsqueeze(1)
    
    log_energy_combined = torch.cat((
        log_energy_frames,
        dft_500Hz_log_energy[:,:,:frame_count],
        magnitude_sorted_dft_log_energy[:,:,:frame_count],
        cepstrum_log_energy[:,:,:frame_count],
        envelope_follower_log_energy[:,:,:frame_count],
        time_domain_low_passed_log_energy
        ),dim=1)
    # visualize the representation
    data = log_energy_combined[0,:,:].cpu().numpy()

    # Plot the spectrogram-like representation
    # plt.figure(figsize=(10, 6))
    # plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.xlabel('Frame idx')
    # plt.ylabel('Frequency Dimension')
    # plt.title('Feature Representation')
    # # Save the figure with high resolution
    # plt.savefig('/home/hmeng/git/ambisonic-acoustic-estimaton/CNN_Volume_feature_representation.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return log_energy_combined

class SimplePowerVector(nn.Module):
    def __init__(self, 
                 params: BandingParams, 
                 fs: int, 
                 nch: int, 
                 cov_to_pv_trainable: bool=False,
                 band_matrix_trainable: bool=False,
                 smoothing_trainable: bool=False,
                 normalise: bool=False, 
                 hz_s_per_band: float=None,
                 stack_bands: bool=False):
        super().__init__()
        # params = BandingParams.Log(dt_ms, fs, BandingShape.SOFT, TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.LPF)
        # params = BandingParams.Mel(dt_ms=dt_ms, fmin=300, fmax=fs // 2, nband=55, shape=BandingShape.TRI, transform_params=TransformParams.RaisedSine(), lower_band_mode=LowerBandMode.HPF)
        dt_ms = params.dt_ms
        # Here is where you would change GPV flavour and hz per band
        spat_params = SpatialBandingParams.from_banding_params(params, hz_s_per_band=hz_s_per_band) 
        coefs = SpatialBandingCoefs(spat_params, fs, dt_ms=dt_ms, nch=nch)
        self.power_vector = PowerVector(
            coefs, 
            nch=nch, 
            cov_to_pv_trainable=cov_to_pv_trainable,
            band_matrix_trainable=band_matrix_trainable,
            smoothing_trainable=smoothing_trainable,
            normalise=normalise, 
            do_phase_adjust=False
        )

        self.reblocker = Reblocker(coefs.block_size) # 320 samples per block
        self.forward_transform = ForwardTransform(params.transform_params, SignalPathConfig(fs, dt_ms, nmic=nch))
        self.stack_bands = stack_bands
        self.custom_normalise = True

        def _remove_tensor_names_recursively(module: nn.Module):
            # First, handle immediate attributes of the current module
            for key, value in module.__dict__.items():
                if isinstance(value, torch.Tensor) and not isinstance(value, nn.Parameter):
                    if any(value.names):
                        # Use setattr to modify the attribute in place
                        setattr(module, key, value.rename(None))

            # Then, descend into child modules
            for child in module.children():
                _remove_tensor_names_recursively(child)

        # Handle registered parameters and buffers
        for p in self.parameters():
            if any(p.names):
                p.data = p.data.rename(None)
        for b in self.buffers():
            if any(b.names):
                b.data = b.data.rename(None)

        # Handle any other tensors stored as attributes recursively
        _remove_tensor_names_recursively(self)
    
    def forward(self, pcm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.forward_transform.hist.state = None
            if hasattr(self.power_vector, "smooth"):
                self.power_vector.smooth.state = None
            pcm = pcm.refine_names("batch" if len(pcm.shape) > 2 else ..., "sample", "ch")
            frames = self.reblocker(pcm) #[84, 760, 128, 4]
            # --> [64,200,320,4]
            bins = self.forward_transform(frames) #[84,128,4,760]
        # import pdb
        # pdb.set_trace()
        pv = self.power_vector(bins) # (B, F, T, C) --> [128,54,84,16]
        if self.custom_normalise: # True
            pv_lvls = torch.clamp(pv[..., :1], 1e-10)
            pv = torch.cat([torch.log(pv_lvls), pv[..., 1:] / pv_lvls], dim=-1)
            
            # pv = torch.cat([pv[..., :1], pv[..., 1:] / pv_lvls], dim=-1)
            # pv = torch.cat([torch.log(pv_lvls), pv[..., :1], pv[..., 1:] / pv_lvls], dim=-1)
            # pv = pv[..., 1:] / pv_lvls
        pv = pv.align_to(..., "frame", "pv", "band") # (B, T, C, F)
        
        if self.stack_bands:
            bins = bins.align_to(..., "frame", "ch", "bin").rename(None)
            bands = torch.matmul(bins * bins.conj(), torch.view_as_complex(self.power_vector.band_matrix)).rename(None).real
            pv = torch.cat([pv.rename(None), bands], dim=-2)

        return pv #[64,200,16,48]

# room transformer feature
def transformer_room_feature(audio, sample_rate=16000):
    """
    Converts a batch of audio signals into Gammatone filterbank and phase spectrogram representations.
    Args:
        audio_batch (torch.Tensor): Tensor of shape [batch, samples].
        sample_rate (int): Sampling rate of the audio.
        n_fft (int): Number of FFT components.
        n_freqs (int): Number of Gammatone filterbank frequencies.
    Returns:
        gammatone (torch.Tensor): Gammatone filterbank features of shape [batch, n_freqs, time].
        phase_spectrogram (torch.Tensor): Phase spectrogram of shape [batch, n_fft//2+1, time].
    """
    # Gammatone filterbank representation
    device = audio.device
    w_audio = audio[:,:,0] # take the w channel onl
    gammatone_transform = torchaudio.transforms.Gammatone(n_freqs=n_freqs, sample_rate=sample_rate)
    gammatone = gammatone_transform(w_audio)
    
    # Phase spectrogram using STFT
    stft = _stft_safe(w_audio, n_fft=n_fft, return_complex=True)
    phase_spectrogram = torch.angle(stft)
    
    return gammatone, phase_spectrogram
