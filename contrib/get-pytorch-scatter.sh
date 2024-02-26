source "$(dirname "$0")/get-script-header.sh"
PROJDIR="$(pwd)"
rm -rf pt_scatter
mkdir pt_scatter
cd "pt_scatter"
git clone https://github.com/rusty1s/pytorch_scatter.git
cd "pytorch_scatter"
#git checkout 2f447cfb282a3e1f803e27a9b61abec19b805d65
mkdir build
cd build
# Might have to install sudo apt-get install python3.7-dev
cmake -DCMAKE_PREFIX_PATH="$PROJDIR/deps/pytorch/libtorch" -DWITH_CUDA=on ..
make DESTDIR="$PROJDIR/deps/pytorch-scatter" install


sed -i 's/#include <torch\/extension.h>/#include <torch\/all.h>/g' "$PROJDIR/deps/pytorch-scatter/usr/local/include/torchscatter/scatter.h"
