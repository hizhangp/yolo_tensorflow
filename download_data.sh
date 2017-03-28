echo "Creating data directory..."
mkdir -p data && cd data

echo "Downloading Pascal VOC 2012 data..."
wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

echo "Extracting VOC data..."
tar xf VOCtrainval_11-May-2012.tar

echo "Done."
