source "$(dirname "$0")/get-script-header.sh"

PYTORCH_URL="https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcu113.zip"
PYTORCH_DIR="$DEPS_DIR/pytorch"
echo "Setting up $PYTORCH_DIR ..."
rm -rf "$PYTORCH_DIR"
mkdir -p "$PYTORCH_DIR"
cd "$PYTORCH_DIR"
webget "$PYTORCH_URL" archive.zip
unzip archive.zip
rm archive.zip