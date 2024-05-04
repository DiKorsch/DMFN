import argparse
import numpy as np

from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('root', help='path to the dataset')
parser.add_argument('output', help='path to the file list')
parser.add_argument('--images_folder', default="images", help='relative path from root to the images')



def main(args):
    root = Path(args.root)
    output = Path(args.output)
    flist = np.loadtxt(root / "images.txt", dtype=[("id", "U255"), ("fname", "U255")])
    splitIDs = np.loadtxt(root / "tr_ID.txt", dtype=np.int32)

    for subset, splitID in [("train", 1), ("val", 0)]:
        mask = splitIDs == splitID

        with open(output / f"images_{subset}.txt", "w") as f:
            for fname in flist[mask]["fname"]:
                print(root / args.images_folder / fname, file=f)


main(parser.parse_args())
