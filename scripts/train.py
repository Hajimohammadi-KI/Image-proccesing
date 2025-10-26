import sys

from xai_proj_b.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
