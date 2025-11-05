#The following code provides the functions that are necessary in order to import and input spectra from the DESI Data Release 1 dataset. It assumes the foundation model-aion has been compiled (which we do in our main main script)
#Import libraries
import torch
import numpy as np
from tqdm import tqdm
from astropy.table import Table

# AION toklenizer
from aion.codecs import CodecManager
codec_manager = CodecManager(device="cuda")

# AION embedder
from aion.model import AION
# Disable gradients for this notebook
torch.set_grad_enabled(False)
model = AION.from_pretrained("polymathic-ai/aion-base").to("cuda").eval()
#model = AION.from_pretrained("polymathic-ai/aion-base").to("cpu").eval()

from aion.modalities import (
    LegacySurveyImage,
    DESISpectrum,
    LegacySurveyFluxG,
    LegacySurveyFluxR,
    LegacySurveyFluxI,
    LegacySurveyFluxZ,
    Z,
)

#Loading the DESI DR1 data. These have also been uploaded to our repository.
import numpy as np
DESIwave = np.load('DESI_wavelengths.npy')
DESIivar = np.load('DESI_ivar.npy')
DESImask = np.load('DESI_mask.npy')
DESIflux = np.load('DESI_flux.npy')

#Reshape wavelength data to desired format to harmonize with the rest
DESIwave_2D = np.tile(DESIwave, (DESIflux.shape[0], 1))

# Helper function
def to_tensor(data_array, dtype="float32"):
    return torch.tensor(np.array(data_array).astype(dtype), device="cuda")
    
#Define new modelity to load these DESI DR1 spectra. Note the slight change in how it's called compared to the function of the main tutorial

def format_data_modalities_spc(i,batch_size, device="cuda"):
    """Formats the input data into modality objects."""

    # Helper function
    #def to_tensor(data_array, dtype="float32"):
    #    return torch.tensor(np.array(data_array).astype(dtype), device=device)


    # Create spectrum modality
    spectrum = DESISpectrum(
        flux=to_tensor(DESIflux[i : i + batch_size,:]),
        ivar=to_tensor(DESIivar[i : i + batch_size,:]),
        mask=to_tensor(DESImask[i : i + batch_size,:].astype(bool), dtype="bool"),
        wavelength=to_tensor(DESIwave_2D[i : i + batch_size,:]),
    )


    return spectrum
    
    
#For ~1300 DESI DR1 spectra, we need 41 batches of 32 size each

batch_size = 32

num_blocks = len(model.encoder)

#New functio for embeddings
def get_embeddings_layer_sp(batch_size):


    sp_embeddings = {blk:[] for blk in range(num_blocks + 1)}


    # Loop through the table in batches
    for i in tqdm(range(0, 41, batch_size)):

        spectrum = format_data_modalities_spc(i,batch_size, device="cuda")

        encoder_tokens_sp, encoder_emb_sp, encoder_mask_sp, _ = model.embed_inputs(
            codec_manager.encode(spectrum), mask=None, num_encoder_tokens=300
        )
        x_sp = encoder_tokens_sp + encoder_emb_sp



        
        for b, blk in enumerate(model.encoder):
            #x_im = blk(x_im, mask=encoder_mask_im)
            #im_embeddings[b].append(model.encoder_norm(x_im.mean(axis=1)))

            x_sp = blk(x_sp, mask=encoder_mask_sp)
            sp_embeddings[b].append(model.encoder_norm(x_sp.mean(axis=1)))



        x_sp = model.encoder_norm(x_sp)
        context_sp = model.decoder_proj_context(x_sp) + encoder_emb_sp
        sp_embeddings[b + 1].append(context_sp.mean(axis=1))

   

    # Concatenate the embeddings from all batches
    for b in range(num_blocks + 1):
        
   
        sp_embeddings[b] = torch.cat(sp_embeddings[b], dim=0).cpu().numpy()


  
    print(f"Successfully processed {DESIflux.shape[0]} images in batches of {batch_size}.")

    return sp_embeddings


#And then the following command returns the embeddings as usual, with only the batch size provided as input. (now commented out)
#sp_embeddings= get_embeddings_layer_sp(batch_size)

