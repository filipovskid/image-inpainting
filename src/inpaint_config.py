from utils.config import BaseConfig


# Default configuration for DCGAN image inpainting
_dcgan_inpaint_config = BaseConfig()

############################################################

# Importance weighting term window size
_dcgan_inpaint_config.w_size = 7

# Back-propagation iteration count
_dcgan_inpaint_config.iter = 1500

# Default configuration for StyleGAN image inpainting
_stylegan_inpaint_config = BaseConfig()

############################################################

#
_stylegan_inpaint_config.alpha = 1

#
_stylegan_inpaint_config.style_weight = 0.7

# Importance weighting term window size
_stylegan_inpaint_config.w_size = 28

# Back-propagation iteration count
_stylegan_inpaint_config.iter = 3000
