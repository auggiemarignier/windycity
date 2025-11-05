#Data massaging for audio spectra from turbulent collisions

import numpy as np  
from aion.modalities import (
    LegacySurveyImage,
    DESISpectrum,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
    Z,
)

# Helper function
def to_tensor(data_array, device="cuda", dtype="float32"):
    return torch.tensor(np.array(data_array).astype(dtype), device=device)

#main function for audio data ingestion
def turbulent_collision_ingestion(astro_data, audio_spectra_raw, device="cuda"):
  
  #load data
  #audio_spectra_raw = np.load('../data/2024_08_28_t0116_c0167_gsw0p84_rpm1200_setup1_camera3_capsule_only_audio_spectra_normalized.npy')
  
  #reshape and normalize
  data_subset = np.copy(audio_spectra_raw)[0:6].reshape(-1, audio_spectra_raw.shape[-1])
  data_subset *= -1
  data_subset -= np.min(data_subset)
  data_subset /= np.max(data_subset)
  
  #get length of spectra expected by astra data
  astro_spectra_length = astro_data["desi_spectrum_lambda"].shape[1]
  
  #loop for stretching spectra to length of spectra expected by astro data
  audio_spectra = []
  for i in np.arange(astro_data["desi_spectrum_lambda"].shape[0]):
    
    audio_spectrum_fnc = interp1d(np.linspace(0,1,num = data_subset.shape[1], endpoint=True), data_subset[i])
    interp_grid = np.linspace(0, 1, astro_spectra_length, endpoint = True)
    audio_spectra.append(audio_spectrum_fnc(interp_grid))
  audio_spectra = np.array(audio_spectra)

  # Create spectrum modality
  spectrum = DESISpectrum(
      flux=to_tensor(audio_spectra),
      ivar=to_tensor(5.*np.ones(astro_spectra_length)),
      mask=to_tensor(np.zeros(astro_spectra_length), dtype="bool"),
      wavelength=to_tensor(astro_data["desi_spectrum_lambda"]),
  )

  return spectrum
