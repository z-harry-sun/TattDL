# TattDL
Tattoo detection and localization

## Caffe Version
This uses the Caffe hash: 4115385deb3b907fcd428ac0ab53b694d741a3c4

## Getting the model
The tattoo model is managed by git-lfs, so make sure that is installed (see
`https://git-lfs.github.com/` for details on how to do that).

Then, fetch the file using:

  git lfs fetch
  git lfs checkout

## Detection output format
After running the `` tool, one of the output files will be a `detection.txt` file.
This is a CSV-like file (`|` primary separators):

  filename | proc time | scale | scores | boxes

The boxes field (the last one) may container 0 or more confidence and bounding box specifications.  Separating this text field by spaces (` `) will yield one or more sub-CSV rows with the format:

  confidence,x,y,width,height

X and Y coordinates specify the upper left corner of the sub-region, assuming (0,0) is the upper left corner of the image.
