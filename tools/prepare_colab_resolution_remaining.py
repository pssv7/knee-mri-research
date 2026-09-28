"""Generate the standalone 384-only remaining-fold notebook; never launch training."""
from tools.prepare_colab_resolution import build

if __name__ == '__main__':
    build(remaining=True)
