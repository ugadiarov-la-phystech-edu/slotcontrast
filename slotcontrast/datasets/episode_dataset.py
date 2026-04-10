import time
from unittest import result

import numpy as np
from torch.utils.data import Dataset
import glob
import os
import os.path as osp
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EpisodesDataset(Dataset):
    def __init__(self, root, mode, transforms=None, extension='png', kind='video', sequence_length=1,
                 mask_extension='png', mask_type=None):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode in ('valid', 'test'):
            mode = 'val'

        assert kind in ('image', 'video'), f'Expected kind: image or video. Actual: {kind}'
        if kind == 'image':
            assert sequence_length == 1, f'Expected sequence length: 1. Actual: {sequence_length}'

        self.kind = kind
        self.sequence_length = sequence_length
        self.transforms = transforms

        root = os.path.join(root, mode)
        root_with_obs = os.path.join(root, 'obs')
        if os.path.isdir(root_with_obs):
            self.root = root_with_obs
        else:
            self.root = root

        # mask_type selects which mask folder to use: masks_class/, masks_instance/, or masks/
        if mask_type is not None:
            masks_root = os.path.join(root, f'masks_{mask_type}')
        else:
            masks_root = os.path.join(root, 'masks')
        self.has_masks = os.path.isdir(masks_root)
        self.masks_root = masks_root if self.has_masks else None
        self.mask_extension = mask_extension

        self.mode = mode
        self.extension = extension

        # Get all numbers
        self.folders = []
        start = time.time()
        for file in os.listdir(self.root):
            try:
                self.folders.append(file)
            except ValueError:
                continue

        def get_num(x):
            parts = x.split('_')
            num = parts[0] if len(parts) == 1 else parts[1]
            return int(num)

        self.folders.sort(key=get_num)

        self.episode_images = []
        self.episode2offset = [0]
        self.index2episode = []
        for i, f in enumerate(self.folders):
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, f'*.{self.extension}')))
            actual_length = len(paths)
            get_file_id = lambda x: get_num(osp.splitext(osp.basename(x))[0])
            paths.sort(key=get_file_id)
            self.episode_images.append(paths)
            self.index2episode.extend([len(self.episode_images) - 1] * actual_length)
            self.episode2offset.append(self.episode2offset[-1] + actual_length)

        print(f'Dataset indexing took {time.time() - start} seconds')

    def _load_mask(self, image_path):
        """Load a dense segmentation mask corresponding to an image path.

        Masks are stored as grayscale PNGs where each pixel value is an instance ID
        (0 = background). The mask file is located by mirroring the image path
        structure under the masks/ directory.
        """
        episode_name = os.path.basename(os.path.dirname(image_path))
        frame_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(
            self.masks_root, episode_name, f'{frame_name}.{self.mask_extension}'
        )
        # Load as grayscale to get integer IDs
        mask = np.array(Image.open(mask_path).convert('L'))
        return mask

    def __getitem__(self, index):
        if self.kind == 'video':
            episode_images = self.episode_images[index]
            start_index = np.random.randint(0, len(episode_images) - self.sequence_length + 1)
            image_sequence = []
            mask_sequence = []
            for image_index in range(start_index, start_index + self.sequence_length):
                img = np.array(Image.open(episode_images[image_index]))
                image_sequence.append(img)
                if self.has_masks:
                    mask_sequence.append(self._load_mask(episode_images[image_index]))

            data = np.stack(image_sequence)
        elif self.kind == 'image':
            ep = self.index2episode[index]
            # Implement continuous indexing
            offset = self.episode2offset[ep]
            in_episode_index = index - offset
            data = np.array(Image.open(self.episode_images[ep][in_episode_index]))
            if self.has_masks:
                mask_sequence = [self._load_mask(self.episode_images[ep][in_episode_index])]
        else:
            assert False, 'Cannot happen!'

        data = {'__key__': str(index), self.kind: data}

        if self.has_masks:
            # Dense integer masks: (F, H, W) for video, (H, W) for image
            # Add trailing dim of 1 for consistency with other datasets: (..., H, W, 1)
            masks = np.stack(mask_sequence)
            if self.kind == 'image':
                masks = masks[0]
            data['segmentations'] = masks[..., np.newaxis]

        if self.transforms:
            for name, transform in self.transforms.items():
                if name in data:
                    data[name] = transform(data[name])

        return data

    def __len__(self):
        if self.kind == 'image':
            return len(self.index2episode)
        elif self.kind == 'video':
            return len(self.episode_images)
        else:
            assert False, 'Cannot happen!'
