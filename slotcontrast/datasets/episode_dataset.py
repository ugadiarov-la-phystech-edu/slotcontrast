import time

import numpy as np
from torch.utils.data import Dataset
import glob
import io
import os
import os.path as osp
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EpisodesDataset(Dataset):
    def __init__(self, source_root, split, transforms=None, extension='png', kind='video', sequence_length=1, episode_folder_pattern='*',
                 cache=False, with_segmentation=False, segmentation_folder='segmentation', segmentation_extension='png'):
        assert split in ['train', 'val', 'valid', 'test']
        if split in ('valid', 'test'):
            split = 'val'

        assert kind in ('image', 'video'), f'Expected kind: image or video. Actual: {kind}'
        if kind == 'image':
            assert sequence_length == 1, f'Expected sequence length: 1. Actual: {sequence_length}'
        if with_segmentation:
            assert kind == 'video', 'Segmentation loading is only supported for kind="video".'

        self.kind = kind
        self.sequence_length = sequence_length
        self.transforms = transforms

        self.source_root = os.path.join(source_root, split)
        self.split = split
        self.extension = extension
        self.cache = cache
        self.episode_folder_pattern = episode_folder_pattern
        self.with_segmentation = with_segmentation
        self.segmentation_folder = segmentation_folder
        self.segmentation_extension = segmentation_extension

        # Get all numbers
        self.episode_ids = []
        start = time.time()
        for path in glob.glob(os.path.join(self.source_root, self.episode_folder_pattern)):
            if os.path.isdir(path):
                self.episode_ids.append(os.path.relpath(path, start=self.source_root))

        def get_num(x):
            path = Path(x)
            name = path.parts[-1]
            parts = name.split('_')
            num = parts[0] if len(parts) == 1 else parts[1]
            if len(path.parts) == 1:
                return int(num)

            return os.path.join(*path.parts[:-1]), int(num)

        self.episode_ids.sort(key=get_num)

        self.episode_images = []
        self.episode_segmentations = []
        self.episode2offset = [0]
        self.index2episode = []
        get_file_id = lambda x: get_num(os.path.splitext(os.path.basename(x))[0])
        for i, f in enumerate(self.episode_ids):
            dir_name = os.path.join(self.source_root, f)
            paths = list(glob.glob(osp.join(dir_name, f'*.{self.extension}')))
            actual_length = len(paths)
            paths.sort(key=get_file_id)
            self.episode_images.append(paths)
            self.index2episode.extend([len(self.episode_images) - 1] * actual_length)
            self.episode2offset.append(self.episode2offset[-1] + actual_length)

            if self.with_segmentation:
                seg_dir = os.path.join(dir_name, self.segmentation_folder)
                if not os.path.isdir(seg_dir):
                    raise FileNotFoundError(
                        f'Segmentation folder not found for episode {f}: {seg_dir}'
                    )
                seg_paths = list(glob.glob(osp.join(seg_dir, f'*.{self.segmentation_extension}')))
                seg_paths.sort(key=get_file_id)
                if len(seg_paths) != actual_length:
                    raise ValueError(
                        f'Episode {f} has {actual_length} frames but {len(seg_paths)} '
                        f'segmentation masks; they must be aligned one-to-one.'
                    )
                self.episode_segmentations.append(seg_paths)

        print(f'Dataset indexing took {time.time() - start} seconds')

        if self.cache:
            # read image files into episode_images as bytes
            start = time.time()
            episode_image_bytes = []
            for paths in tqdm(self.episode_images, desc=f'Caching split: {self.split}'):
                image_bytes = []
                for path in paths:
                    with open(path, 'rb') as f:
                        image_bytes.append(io.BytesIO(f.read()))

                episode_image_bytes.append(image_bytes)

            self.episode_images = episode_image_bytes

            if self.with_segmentation:
                episode_segmentation_bytes = []
                for paths in tqdm(self.episode_segmentations, desc=f'Caching segmentations: {self.split}'):
                    seg_bytes = []
                    for path in paths:
                        with open(path, 'rb') as f:
                            seg_bytes.append(io.BytesIO(f.read()))

                    episode_segmentation_bytes.append(seg_bytes)

                self.episode_segmentations = episode_segmentation_bytes
            print(f'Dataset caching took {time.time() - start} seconds')

    def __getitem__(self, index):
        segmentation = None
        if self.kind == 'video':
            episode_images = self.episode_images[index]
            start_index = np.random.randint(0, len(episode_images) - self.sequence_length + 1)
            image_sequence = []
            for image_index in range(start_index, start_index + self.sequence_length):
                img = np.array(Image.open(episode_images[image_index]))
                image_sequence.append(img)

            data = np.stack(image_sequence)

            if self.with_segmentation:
                episode_segmentations = self.episode_segmentations[index]
                mask_sequence = []
                for image_index in range(start_index, start_index + self.sequence_length):
                    mask = np.array(Image.open(episode_segmentations[image_index]))
                    mask_sequence.append(mask)
                # (F, H, W) -> (F, H, W, 1): trailing channel expected by `ToTensorMask`.
                segmentation = np.stack(mask_sequence)[..., None]
        elif self.kind == 'image':
            ep = self.index2episode[index]
            # Implement continuous indexing
            offset = self.episode2offset[ep]
            in_episode_index = index - offset
            data = np.array(Image.open(self.episode_images[ep][in_episode_index]))
        else:
            assert False, 'Cannot happen!'

        data = {'__key__': str(index), self.kind: data}
        if segmentation is not None:
            data['segmentations'] = segmentation

        if self.transforms:
            for name, transform in self.transforms.items():
                data[name] = transform(data[name])

        return data

    def __len__(self):
        if self.kind == 'image':
            return len(self.index2episode)
        elif self.kind == 'video':
            return len(self.episode_images)
        else:
            assert False, 'Cannot happen!'
