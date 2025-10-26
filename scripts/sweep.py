import sys

from xai_proj_b.cli import main

if __name__ == "__main__":
    main(["sweep", *sys.argv[1:]])
