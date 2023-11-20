mkdir -p /tmp/imagenet
cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/imagenet/val.zip /tmp/imagenet/val.zip
pushd /tmp/imagenet/
mkdir val
cd val/
unzip -qq ../val.zip
popd

