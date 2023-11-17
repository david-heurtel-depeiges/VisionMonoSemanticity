cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet /tmp/imagenet
pushd /tmp/imagenet/
mkdir val train
cd train/
unzip -qq ../train.zip
cd ../val/
unzip -qq ../val.zip
popd
