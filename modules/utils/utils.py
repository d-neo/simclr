import gdal
import osgeo.gdalnumeric as gdn
import numpy as np


def get_map_extent(gdal_raster):
    """Returns a dict of {xmin, xmax, ymin, ymax, xres, yres} of a given GDAL raster file.

    Args:
        gdal_raster: File opened via gdal.Open().
    """
    xmin, xres, xskew, ymax, yskew, yres  = gdal_raster.GetGeoTransform()
    xmax = xmin + (gdal_raster.RasterXSize * xres)
    ymin = ymax + (gdal_raster.RasterYSize * yres)
    # ret = ( (ymin, ymax), (xmin, xmax) )
    ret = {"xmin":xmin, "xmax":xmax, "ymin":ymin, "ymax":ymax, "xres":xres, "yres":yres}
    if 0. in (ymin, ymax, xmin, xmax): return None # This is true, if no real geodata is referenced.
    return ret


def img_to_array(input_file, dim_ordering="HWC", dtype='float32', band_mapping=None, return_extent=False):
    """Reads an image from disk and returns it as numpy array.

    Args:
        input_file: Path to the input file.
        dim_ordering: One of HWC or CHW, C=Channels, H=Height, W=Width
        dtype: Desired data type for loading, e.g. np.uint8, np.float32...
        band_mapping: Dictionary of which image band to load into which array band. E.g. {1:0, 3:1}
        return_extent: Whether or not to return the raster extent in the form (ymin, ymax, xmin, xmax). Defaults to False.

    Returns:
        Numpy array containing the image and optionally the extent.
    """
    ds = gdal.Open(input_file)
    extent = get_map_extent(ds)

    if band_mapping is None:
        num_bands = ds.RasterCount
        band_mapping = {i+1: i for i in range(num_bands)}
    elif isinstance(band_mapping, dict):
        num_bands = len(band_mapping)
    else:
        raise TypeError("band_mapping must be a dict, not {}.".format(type(band_mapping)))

    arr = np.empty((num_bands, ds.RasterYSize, ds.RasterXSize), dtype=dtype)

    for source_layer, target_layer in band_mapping.items():
        arr[target_layer] = gdn.BandReadAsArray(ds.GetRasterBand(source_layer))

    if dim_ordering == "HWC":
        arr = np.transpose(arr, (1, 2, 0))  # Reorders dimensions, so that channels are last
    elif dim_ordering == "CHW":
        pass
    else:
        raise ValueError("Dim ordering {} not supported. Choose one of 'HWC' or 'CHW'.".format(dim_ordering))

    if return_extent:
        return arr, extent
    else:
        return arr


def array_to_tif(array, src_raster, dst_filename, num_bands='multi', save_background=True):
    """ Takes a numpy array and writes a tif. Uses deflate compression.

    Args:
        array: numpy array
        src_raster (str): Raster file used to determine the corner coords.
        dst_filename (str): Destination file name/path
        num_bands (str): 'single' or 'multi'. If 'single' is chosen, everything is saved into one layer. The values
            in each layer of the input array are multiplied with the layer index and summed up. This is suitable for
            mutually exclusive categorical labels or single layer arrays. 'multi' is for normal images.
    """
    src_raster = gdal.Open(src_raster)
    x_pixels = src_raster.RasterXSize
    y_pixels = src_raster.RasterYSize

    bands = min( array.shape ) if array.ndim==3 else 1
    if not save_background and array.ndim==3: bands -= 1

    driver = gdal.GetDriverByName('GTiff')

    datatype = str(array.dtype)
    datatype_mapping = {'byte' : gdal.GDT_Byte, 'uint8' : gdal.GDT_Byte, 'uint16': gdal.GDT_UInt16,
                        'uint32': gdal.GDT_UInt32, 'int8':gdal.GDT_Byte, 'int16':gdal.GDT_Int16,
                        'int32':gdal.GDT_Int32, 'float16': gdal.GDT_Float32, 'float32': gdal.GDT_Float32}
    options = ["COMPRESS=DEFLATE"]
    if datatype == "float16":
        options.append("NBITS=16")

    out = driver.Create(
        dst_filename,
        x_pixels,
        y_pixels,
        1 if num_bands == 'single' else bands,
        datatype_mapping[datatype],
        options=options)

    out.SetGeoTransform( src_raster.GetGeoTransform() )
    out.SetProjection(   src_raster.GetProjection()   )

    if array.ndim == 2:
        out.GetRasterBand(1).WriteArray(array)
        out.GetRasterBand(1).SetNoDataValue(0)
    else:
        if num_bands == 'single':
            singleband = np.zeros(array.shape[:2], dtype=array.dtype)
            for i in range(bands):
                singleband += (i+1)*array[:,:,i]
            out.GetRasterBand(1).WriteArray( singleband )
            out.GetRasterBand(1).SetNoDataValue(0)

        elif num_bands == 'multi':
            for i in range(bands):
                out.GetRasterBand(i+1).WriteArray( array[:,:,i] )
                out.GetRasterBand(i+1).SetNoDataValue(0)

    out.FlushCache()  # Write to disk.

def percentile(arr, low=1, high=99, mode=None, no_data_vec=None, estimate=True, copy=False):
    """Balances colors either channelwise or for the whole image.

    out = (arr - p(low))/(p(high)-p(low))
    where p is the percentile value.

    p(99) is the value, for which 99% of pixels are darker than this value.

    Out values are clipped to (0,1).

    WARNING: Layerwise modifies in-place!

    Args:
        arr (ndarray): Input array.
        low (int or float): Low percentile value, default: 1
        high (int or float): High percentile value, default: 99
        mode (str): 'layerwise' will treat each layer individually
        no_data_vec (iterable): Regions equaling this color will be omitted in color balancing.
        estimate (bool): If True, percentile value will be estimated using 1/16th of the data.
        copy (bool): If True, acts on a copy of the data. Otherwise the input array is modified!

    Returns: Color balanced array.
    """

    inc = 4 if estimate else 1
    shape = arr.shape
    if no_data_vec is not None:
        assert shape[-1]==len(no_data_vec), "Length of no_data_vec must match number of channels."
        data_mask = np.all(arr.reshape( (-1,shape[-1]) ) != no_data_vec, axis=1).reshape(shape[:2])[::inc, ::inc]

    if mode=='layerwise' and arr.ndim==3:
        img = arr if not copy else np.copy(arr)
        for i in range(arr.shape[-1]):
            tmp1 = arr[::inc,::inc,i]
            tmp2 = tmp1[data_mask] if no_data_vec is not None else tmp1
            minp = np.percentile(tmp2, low)
            maxp = np.percentile(tmp2, high) - minp + 1E-4
            img[:,:,i] = np.clip((arr[:,:,i] - minp) / maxp, 0, 1)
        return img

    else:
        tmp1 = arr[::inc,::inc]
        tmp2 = tmp1[data_mask] if no_data_vec is not None else tmp1
        minp = np.percentile(tmp2, low)
        maxp = np.percentile(tmp2, high) - minp + 1E-4
        return np.clip((arr - minp) / maxp, 0, 1)