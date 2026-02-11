# Autoencoder-based
from .EASE import EASE
from .MultVAE import MultVAE

# MF
from .MF import MF

# LOCA
from .LOCA_VAE import LOCA_VAE
from .LOCA_EASE import LOCA_EASE

# MOE
from .MOE import MOE

# weighted loss vae
from .WL import WL

from .R_MOE import R_MOE

# popularity debiasing baseline
from .MF_BPR import MF_BPR
from .MACR import MACR
from .BC_Loss import BC_Loss
from .Zerosum import Zerosum

from .MF_boost import MF_boost
from .MF_adaboost import MF_adaboost
from .vae_adaboost import vae_adaboost

from .neu_rec import NeuralRecommender

from .autoencoder import autoencoder

from .rrl import RRL