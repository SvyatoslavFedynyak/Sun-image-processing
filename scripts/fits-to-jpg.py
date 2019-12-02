import os
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style

plt.style.use(astropy_mpl_style)

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

for file in os.listdir('.'):

    if not file.endswith(".fits"):
        continue

    # Open
    print(file)
    image_file = get_pkg_data_filename(file)
    image_data = fits.getdata(image_file, ext=0)

    # Save

    plt.figure()
    plt.imshow(image_data)
    plt.axis('off')
    file=file.replace('fits', 'jpg', 1)
    plt.savefig('../jpg/{0}'.format(file), bbox_inches='tight', dpi=300)
    plt.close('all')
