# Set up matplotlib and use a nicer set of plot parameters

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

# Download the example FITS files used by this example:

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

image_file = get_pkg_data_filename('PIC-SOD-20100806_0107-535D-RS-MNM-L1B-SL4R_20140604-O1_D1_C0_R1_G1_F0_X0_P0.fits')

# Use `astropy.io.fits.info()` to display the structure of the file:

fits.info(image_file)

# Generally the image information is located in the Primary HDU, also known
# as extension 0. Here, we use `astropy.io.fits.getdata()` to read the image
# data from this first extension using the keyword argument ``ext=0``:

image_data = fits.getdata(image_file, ext=0)

# The data is now stored as a 2D numpy array. Print the dimensions using the
# shape attribute:

print(image_data.shape)

# Display the image data:

plt.figure()
plt.imshow(image_data)
plt.savefig('PIC-SOD-20100806_0107-535D-RS-MNM-L1B-SL4R_20140604-O1_D1_C0_R1_G1_F0_X0_P0.png')
plt.savefig('PIC-SOD-20100806_0107-535D-RS-MNM-L1B-SL4R_20140604-O1_D1_C0_R1_G1_F0_X0_P0.tiff')
plt.savefig('PIC-SOD-20100806_0107-535D-RS-MNM-L1B-SL4R_20140604-O1_D1_C0_R1_G1_F0_X0_P0.jpg', bbox_inches='tight')
plt.colorbar()
