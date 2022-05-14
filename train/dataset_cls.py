from PIL import Image
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
import torch
import utils
import openslide

Image.MAX_IMAGE_PIXELS = 5000000000
SVS_LEVEL = 1

class BaseDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            opt: Any,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms
        self.opt = {}

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""


class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def default_loader_2d(path: str) -> Any:
    #return nii_loader(path)
    return pil_loader(path)
def svs_loader_2d(path: str) -> Any:
    return svs_loader(path)

class DatasetList2d(BaseDataset):
    def __init__(
            self,
            root: str,
            opt: Any,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            extensions: Optional[Tuple[str, ...]] = None,
            loader: Callable[[str], Any] = default_loader_2d,
            loader_svs: Callable[[str], Any] = svs_loader_2d,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetList2d, self).__init__(root, opt, transform=transform,
                                            target_transform=target_transform)

        # _classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}
        # class_to_idx = _classes
        # classes = [f for f in class_to_idx.keys()]
        class_to_idx = {name: i for i, name in enumerate(opt.md_classes)}
        samples = self._get_list(root)

        if len(samples) == 0:
            msg = "Found 0 files in list of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.loader_svs = loader_svs
        self.extensions = extensions

        self.classes = opt.md_classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.root_dataset = opt.root_dataset
        self.stain_norm = opt.stain_norm
        # self.targets = [s[1] for s in samples]

        self.stain_norm_dataset = {}
        if opt.stain_norm:
            stain_target_img = np.array(Image.open(opt.stain_norm))
            target = utils.standardize_brightness(stain_target_img)
            self.stain_matrix_target = utils.get_stain_matrix(target)
            print('Processing stain normalization..')

            for idx, sp in enumerate(self.samples):
                print(sp[0])
                # stain normalization
                img = Image.open(sp[0])
                img = self.stain_normalization(img)
                self.stain_norm_dataset[sp[0]] = img
                debug = 0

    def _get_list(self, list_path):
        new_list = []
        f = open(list_path)
        lines = f.readlines()
        for line in lines:
            line = line.splitlines()[0]
            split_line = line.split(' ')
            if len(split_line) == 2 or len(split_line) == 3:
                new_list.append((split_line[0], int(split_line[1])))
            elif len(split_line) == 1:
                new_list.append([split_line[0]])

        return new_list

    def stain_normalization(self, img):
        img = np.array(img)
        img = utils.VahadaneNormalizer_transform(img, self.stain_matrix_target)

        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # print (self.samples[index])
        if len(self.samples[index]) == 2:
            path, target = self.samples[index]
        elif len(self.samples[index]) == 1:
            path = self.samples[index][0]
            self.target_transform = None
            target = -1
        else:
            raise IOError('Error: Loading samples')


        ### stain normalization
        if self.stain_norm:
            sample = self.stain_norm_dataset[path]
            sample = Image.fromarray(sample)

        elif os.path.splitext(path)[1].lower() == '.svs':
            sample = self.loader_svs(path)

            """
            img = self.loader_svs(path)

            sample = []
            targets = []

            cp_size = (512, 512)
            cp_loc, _cp_size = utils.crop_pad(img.size, crop_size=cp_size)
            for c_idx, c_l in enumerate(cp_loc):
                cp_img = img.crop((c_l[0], c_l[1], c_l[0] + _cp_size[0], c_l[1] + _cp_size[1]))
                pad_val = utils.get_pad_val(cp_size, _cp_size)
                if pad_val[0]:
                    cp_img = cp_img.crop((pad_val[1]))

                if self.transform is not None:
                    cp_img = self.transform(cp_img).unsqueeze(0)

                sample.append(cp_img)
                targets.append(target)

            if self.target_transform is not None:
                targets = self.target_transform(targets)
            target = targets
            sample = torch.cat(sample, dim=0)
            """
        else:
            sample = self.loader(path)

            if target == 4 or target == 11 or target == 14:
                target = 0
            elif target == 13:
                target = 1
            elif target == 12:
                target = 2

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def svs_loader(path):
    slide = openslide.OpenSlide(path)
    slide = slide.read_region((0, 0), SVS_LEVEL, slide.level_dimensions[SVS_LEVEL]).convert('RGB')
    return slide

def nii_loader(path):
    import nibabel as nib

    nii_img = nib.load(path)

    nii_img_arr = np.array(nii_img.dataobj)
    nii_img_arr = Get_LUT_value3D(nii_img_arr.transpose((2, 1, 0)), 1500, -600, rescaleIntercept=0, rescaleSlope=1)

    return nii_img_arr

def Get_LUT_value3D(datas, window, level, rescaleIntercept=0, rescaleSlope=1):
    new_img = []
    for data in datas:
        if isinstance(window, list):
            window = window[0]
        if isinstance(level, list):
            level = level[0]
        arr = np.piecewise(data,
                           [((data * rescaleSlope) + rescaleIntercept) <= (level - 0.5 - (window - 1) / 2),
                            ((data * rescaleSlope) + rescaleIntercept) > (level - 0.5 + (window - 1) / 2), ],
                           [0, 255, lambda VAL: ((((VAL * rescaleSlope) + rescaleIntercept) - (level - 0.5)) / (
                                   window - 1) + 0.5) * (255 - 0)])

        new_img.append(arr)

    return np.array(new_img)


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances