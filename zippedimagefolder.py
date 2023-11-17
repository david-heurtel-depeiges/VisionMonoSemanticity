#!/usr/bin/env python
import os
import struct
import sys
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets import DatasetFolder
from zipfile import Path, ZipFile
import json
from io import BytesIO

import cv2
import numpy as np


from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#########################################################################
## Code copied from Geraud Krawezik from SCC @ Flatiron Institute, NYC ##
#########################################################################

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

class ZippedDatasetFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, None, extensions=extensions, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        self.zipped_file = None

    def loader_zipped(self, path):
        #if self.zipped_file == None:
        #    self.zipped_file = ZipFile(self.root)
        #dataEnc = self.zipped_file.open(path)
        #img = Image.open(dataEnc)
        #print("mode, size:", img.mode, img.size)
        #dataEnc = self.zipped_file.read(path[0])
        #img = Image.frombytes(path[1], path[2], dataEnc, decoder_name='jpeg')

        #dataEnc = self.zipped_file.read(path)
        #img = Image.open(BytesIO(dataEnc))

        if self.zipped_file == None:
            self.zipped_file = open(self.root, 'rb')
        self.zipped_file.seek(path[0])
        dataEnc = self.zipped_file.read(path[1])

        img = cv2.imdecode(np.frombuffer(dataEnc, np.uint8), 1)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img
        img = Image.open(BytesIO(dataEnc), format='JPEG')
        return img.convert("RGB")

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)
        #zipped_dir = Path(directory)
        #zipped_file = ZipFile(directory)
        #zipped_fd = open(directory, 'rb')

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        # Check for filesystem.json
        #LOCAL_FILE_HEADER_OFFSET_TO_LENGTHS = 26
        try:
            #toc = open(directory + '.json')
            toc = open(directory + '_offsets.json')
            #toc = zipped_file.open('filesystem.json')
            filesystem_tree = json.loads(toc.read()) 
            for target_class in sorted(class_to_idx.keys()):
                class_index = class_to_idx[target_class]
                for fname in filesystem_tree[target_class]:
                    #if is_valid_file(fname):
                    #    path = os.path.join(target_class, fname)
                    #    item = path, class_index
                    if True: #TODO: keep fname so that we can do that check!
                        item = fname, class_index
        #            if is_valid_file(fname):
        #                path = os.path.join(target_class, fname)
        #                zinfo = zipped_file.getinfo(path)

         #               zipped_fd.seek(zinfo.header_offset + LOCAL_FILE_HEADER_OFFSET_TO_LENGTHS)
         #               lengths_str = zipped_fd.read(4) # file name and extra field lengths
         #               fnl, efl = struct.unpack("hh", lengths_str)
         #               name = zipped_fd.read(fnl)
                        #print(zinfo.header_offset, path, fnl, efl, name)
         #               item = [zinfo.header_offset + LOCAL_FILE_HEADER_OFFSET_TO_LENGTHS + 4 + fnl + efl, zinfo.file_size], class_index
                        #print(f'{zinfo.filename}: offset {zinfo.header_offset} size: {zinfo.file_size} {zinfo.compress_size}')
                    #if is_valid_file(fname[0]):
                    #    path = os.path.join(target_class, fname[0])
                    #    item = [path, fname[1], fname[2]], class_index
                        instances.append(item)
                    if target_class not in available_classes:
                        available_classes.add(target_class)
        except Exception as e:
            print("No or non-parseable filesystem.json", e)

            for target_class in sorted(class_to_idx.keys()):
                class_index = class_to_idx[target_class]
                zipped_dir = Path(directory)
                target_dir = zipped_dir.joinpath(target_class)
                if not target_dir.is_dir():
                    continue
                #for fnames in sorted(target_dir.iterdir()):
                for fnames in target_dir.iterdir():
                    #for fname in sorted(fnames):
                    #for fname in fnames.iterdir():
                    if True:
                        fname = fnames
                        #path = zipped_dir.joinpath(fname)
                        path = fname
                        if is_valid_file(path.name):
                            #TODO: now it is ./nano_.../val/...
                            item = path, class_index
                            instances.append(item)

                            if target_class not in available_classes:
                                available_classes.add(target_class)
        finally:
            #zipped_fd.close()
            empty_classes = set(class_to_idx.keys()) - available_classes
            if empty_classes:
                msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
                if extensions is not None:
                    msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
                raise FileNotFoundError(msg)

        return instances


    def find_classes(self, root: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        #self.zipped_dir = Path(root)
        zipped_dir = Path(root)
        #zipped_file = ZipFile(root)
        # Check for filesystem.json
        try:
            toc = open(root + '_offsets.json')
            #toc = zipped_file.open('filesystem.json')
            filesystem_tree = json.loads(toc.read()) 
            classes = sorted(filesystem_tree.keys())
        except Exception as e:
            print("No filesystem.json or non parseable", e)
            #classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
            #classes = sorted(entry.name for entry in self.zipped_dir.iterdir() if entry.is_dir())
            classes = sorted(entry.name for entry in zipped_dir.iterdir() if entry.is_dir())
            if not classes:
                raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        finally:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]        
        sample = self.loader_zipped(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

#    def __getstate__(self):
#        print("TODO: serialize")
#        state = ""
#        return state

#    def __setstate__(self, state):
#        print("TODO: deserialize")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Need 1 argument: zip file")
        exit(-1)

    zip_file = sys.argv[1]

    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

    zdf = ZippedDatasetFolder(zip_file, extensions=IMG_EXTENSIONS)

