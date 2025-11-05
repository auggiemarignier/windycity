import numpy as np
from scipy.interpolate import interp1d
from aion.modalities import (
    LegacySurveyImage,
    DESISpectrum,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
    Z,
)

#ingestion function to massage audio data into expected format
def turbulent_collision_ingestion(astro_data, audio_spectra_raw, device="cuda"):

  #get length of spectra expected by astro data
  astro_spectra_length = astro_data["desi_spectrum_lambda"].shape[1]

  #loop for stretching spectra to length of spectra expected by astro data
  audio_spectra = []
  for i in np.arange(astro_data["desi_spectrum_lambda"].shape[0]):

    audio_spectrum_fnc = interp1d(np.linspace(0,1,num = audio_spectra_raw.shape[1], endpoint=True), audio_spectra_raw[i])
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
