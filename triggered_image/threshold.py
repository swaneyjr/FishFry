import rawpy
import numpy
import argparse
import pathlib




def hotcells(array, x_res):
    
    hotc = numpy.load(hotcellFile);
    hotc_array = hotc['hot'];
    x = hotc_array % (x_res);
    y = hotc_array // (x_res);

    maskedarray = numpy.ma.masked_array(array);
    maskedarray[y,x] = numpy.ma.masked;
    return numpy.ma.filled(maskedarray,0);

def checkPixels(array,size):
        
    aboveThreshold = numpy.array(numpy.nonzero(array > threshold));

    x = aboveThreshold[1];
    y = aboveThreshold[0];
    luminosities = array[array > threshold];
    data = numpy.column_stack((aboveThreshold[1], aboveThreshold[0], luminosities));
    numberOfDetections = len(luminosities);
    
    if args.out:

        pathName = pathlib.Path(outputDirectory);

        if not pathName.is_dir():
            pathName.mkdir(parents = True);
    
   
    numpy.savez(fileName, x = x, y = y, luminosities = luminosities, data = data, numberOfDetections = numberOfDetections);



def loadImage(name):
    image = rawpy.imread(name);
    array = image.raw_image;
    size = array.shape
    
    if args.hot:
        array = hotcells(array, size[1]);
    
    checkPixels(array,size);


parser = argparse.ArgumentParser();
parser.add_argument('-f', type = str, nargs = '+', help = "gives location of image");
parser.add_argument('-t', type = int, nargs = 1, help = 'sets threshold', required = True);
parser.add_argument('-out', type = str, nargs = 1, help = 'sets directory for .npz files to be saved into');
parser.add_argument('-hot', type = str, nargs = 1, help = 'gives location of npz file for a hot cell mask');
args = parser.parse_args();

threshold = args.t[0];

if args.hot:
    hotcellFile = args.hot[0];

if args.out:
    outputDirectory = args.out[0];

if args.f:
    for files in args.f:    
        
        fileName = files.split('.')[0] + '.npz';
        
        if args.out:
            fileName = outputDirectory + '/' + pathlib.PurePath(fileName).name;

        loadImage(files);
    


       
