# VOC Kmeans AnchorPoints
The original version from [WillieMaddox](https://github.com/WillieMaddox) is available [here](https://gist.github.com/WillieMaddox/3b1159baecb809b5fcb3a6154bc3cb0b). The purpose of this version is to provide a simpler solution to drop into any VOC project and provide documentation for new users. Coco functionality has been ignored as I did not require it.

# Usage!
For use with darknet download the 2007-2012 datasets, however this library does work with any VOC dataset.

```sh
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```

### On Windows:
Manually donwload each tar file and use winrar,winzip or 7zip to extract the files into a single location.

### python setup
Instance the class in your python project:
```sh
import AnchorLibrary.Anchors as AnchorLibrary


vocLocation = "X:\\Downloads\\training\\VOCdevkit"
getAnchors = AnchorLibrary.Anchor(vocLocation)
```