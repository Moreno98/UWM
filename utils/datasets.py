import os
import json
from torch.utils.data import Dataset
import random
from PIL import Image
import decord

class CustomDataset(Dataset):
    def __init__(self, data, preprocess):
        super().__init__()
        self.data = data
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        incremental_id, safe_image_path, nsfw_image_path, safe_text, unsafe_text, tag = self.data[index]
        safe_image = Image.open(safe_image_path).convert('RGB')
        nsfw_image = Image.open(nsfw_image_path).convert('RGB')
        if self.preprocess:
            if self.preprocess.__class__.__name__ == 'Compose' or self.preprocess.__class__.__name__ == 'ToTensor':
                safe_image = self.preprocess(safe_image)
                nsfw_image = self.preprocess(nsfw_image)
            else:
                safe_image = self.preprocess(images=safe_image, return_tensors="pt")['pixel_values'].squeeze(0)
                nsfw_image = self.preprocess(images=nsfw_image, return_tensors="pt")['pixel_values'].squeeze(0)
        return incremental_id, safe_image, nsfw_image, safe_text, unsafe_text, tag

class ViSU_Text(Dataset):
    def __init__(
        self, 
        dataset_info, 
        split, 
        concept='all', 
        generated_images_path = None, 
        n_generated_images = None, 
        preprocess=None, 
        subset=None, 
        shuffle=False
    ):
        super().__init__()
        assert split in ['train', 'validation', 'test'], 'Split must be one of [train, validation, test]'
        assert concept in dataset_info['concepts']+['all'], f'Concept must be one of {dataset_info["concepts"]+["all"]}'
        self.dataset_path = dataset_info['path']
        self.split = split
        self.concept = concept
        self.preprocess = preprocess
        with open(os.path.join(self.dataset_path, f'ViSU-Text_{self.split}.json'), 'r') as f:
            self.data = json.load(f)
        self.flatten_data = {}
        self.contrastive_prompts = []
        for data in self.data:
            tag = data['tag']
            incremental_id = data['incremental_id']
            if self.concept == 'all' or tag.lower() == self.concept.lower():
                if generated_images_path == None or not self.already_generated(generated_images_path, incremental_id, n_generated_images):
                    safe_text = data['safe']
                    unsafe_text = data['nsfw']
                    self.flatten_data[tag] = self.flatten_data.get(tag, []) + [safe_text, unsafe_text]
                    self.contrastive_prompts.append((incremental_id, safe_text, unsafe_text, tag))
        if shuffle:
            random.shuffle(self.contrastive_prompts)
        self.contrastive_prompts = self.contrastive_prompts[:subset]

    def already_generated(self, generated_images_path, incremental_id, n_generated_images):
        return not os.path.exists(os.path.join(generated_images_path, str(incremental_id))) or not len(os.listdir(os.path.join(generated_images_path, str(incremental_id)))) >= n_generated_images
    
    def get_concept_data(self, concept):
        return [sample for sample in self.contrastive_prompts if sample[-1].lower() == concept.lower() or concept.lower() == 'all']

    def __len__(self):
        return len(self.contrastive_prompts)

    def __getitem__(self, index):
        incremental_id, safe_text, unsafe_text, tag = self.contrastive_prompts[index]
        return incremental_id, safe_text, unsafe_text, tag

    def get_data(self):
        return self.contrastive_prompts
    
    def get_flatten_data(self, concept):
        return self.flatten_data[concept]
    

class ViSU_Full(Dataset):
    def __init__(
        self,
        dataset_info,
        split,
        concept='all',
        preprocess=None,
        subset=None,
        shuffle=False
    ):
        super().__init__()
        assert split in ['train', 'validation', 'test'], 'Split must be one of [train, validation, test]'
        assert concept in dataset_info['concepts']+['all'], f'Concept must be one of {dataset_info["concepts"]+["all"]}'
        self.dataset_path = dataset_info['path']
        self.split = split
        self.concept = concept
        self.preprocess = preprocess
        nsfw_images_path = dataset_info['nsfw_images_path'].format(split)
        with open(os.path.join(self.dataset_path, f'ViSU-Text_{self.split}.json'), 'r') as f:
            self.data = json.load(f)
        self.contrastive_prompts = []

        coco_splits = os.listdir(os.path.join(dataset_info['coco_images_path']))
        self.coco_roots = [os.path.join(dataset_info['coco_images_path'], root) for root in coco_splits]

        for data in self.data:
            tag = data['tag']
            incremental_id = data['incremental_id']
            coco_id = data['coco_id']
            if self.concept == 'all' or tag.lower() == self.concept.lower():
                safe_image_path = self.get_coco_path(coco_id)
                nsfw_image_path = os.path.join(nsfw_images_path, str(incremental_id), '0.jpg')
                safe_text = data['safe']
                unsafe_text = data['nsfw']
                if split == 'train' and shuffle:
                    if os.path.exists(nsfw_image_path):
                        self.contrastive_prompts.append((incremental_id, safe_image_path, nsfw_image_path, str(safe_text), str(unsafe_text), tag))
                else:
                    self.contrastive_prompts.append((incremental_id, safe_image_path, nsfw_image_path, str(safe_text), str(unsafe_text), tag))
        
        if shuffle:
            random.shuffle(self.contrastive_prompts)
        self.contrastive_prompts = self.contrastive_prompts[:subset]

    def get_coco_path(self, coco_id):
        for root in self.coco_roots:
            image_path = os.path.join(root, str(coco_id).zfill(12)+'.jpg')
            if os.path.exists(image_path):
                return image_path
        raise Exception(f'Image not found for coco_id: {coco_id}')

    def get_concept_data(self, concept):
        return [sample for sample in self.contrastive_prompts if sample[-1].lower() == concept.lower() or concept.lower() == 'all']

    def get_sample(self, index):
        return self.contrastive_prompts[index]

    def __len__(self):
        return len(self.contrastive_prompts)

    def __getitem__(self, index):
        incremental_id, safe_image_path, nsfw_image_path, safe_text, unsafe_text, tag = self.contrastive_prompts[index]
        safe_image = Image.open(safe_image_path).convert('RGB')
        nsfw_image = Image.open(nsfw_image_path).convert('RGB')
        if self.preprocess:
            if self.preprocess.__class__.__name__ == 'Compose' or self.preprocess.__class__.__name__ == 'ToTensor':
                safe_image = self.preprocess(safe_image)
                nsfw_image = self.preprocess(nsfw_image)
            else:
                safe_image = self.preprocess(images=safe_image, return_tensors="pt")['pixel_values'].squeeze(0)
                nsfw_image = self.preprocess(images=nsfw_image, return_tensors="pt")['pixel_values'].squeeze(0)
        return incremental_id, safe_image, nsfw_image, safe_text, unsafe_text, tag

    def get_data(self):
        return self.contrastive_prompts
    
class ConceptDataset(Dataset):
    def __init__(self, data, preprocess):
        super().__init__()
        self.data = data
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert len(self.data[index]) == 6, f'Invalid data shape {len(self.data[index])}'
        incremental_id, safe_image_path, nsfw_image_path, safe_text, unsafe_text, tag = self.data[index]
        safe_image = Image.open(safe_image_path).convert('RGB')
        nsfw_image = Image.open(nsfw_image_path).convert('RGB')
        if self.preprocess:
            if self.preprocess.__class__.__name__ == 'Compose':
                safe_image = self.preprocess(safe_image)
                nsfw_image = self.preprocess(nsfw_image)
            else:
                safe_image = self.preprocess(images=safe_image, return_tensors="pt")['pixel_values'].squeeze(0)
                nsfw_image = self.preprocess(images=nsfw_image, return_tensors="pt")['pixel_values'].squeeze(0)
        return incremental_id, safe_image, nsfw_image, safe_text, unsafe_text, tag
    
class UCF_101(Dataset):
    def __init__(self, root, train=False, transform=None):
        self.root_dir = root
        self.train = train
        self.test_lists = ['testlist01.txt',  'testlist02.txt',  'testlist03.txt']
        self.transform = transform
        self._load_videos()
        self.fps = 30

    def _load_videos(self):
        """
        Load the video files from the root directory.
        """
        test_videos = []
        for test_list in self.test_lists:
            with open(os.path.join(self.root_dir, 'ucfTrainTestlist', test_list)) as f:
                test_videos += f.readlines()

        with open(os.path.join(self.root_dir, 'ucfTrainTestlist', 'classInd.txt')) as f:
            class_indices_file = f.readlines()

        # class_indices = [int(line.strip().split()[0])-1 for line in class_indices_file]
        # class_names = [line.strip().split()[1] for line in class_indices_file]
        # classes = os.listdir(os.path.join(self.root_dir, 'UCF-101'))
        # # sort classes
        # classes.sort()

        # self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.class_to_idx = {line.strip().split()[1]: int(line.strip().split()[0])-1 for line in class_indices_file}

        self.videos = []
        for video in test_videos:
            video = video.strip()
            class_name = video.split('/')[0]
            class_idx = self.class_to_idx[class_name]
            if video.endswith(("mp4", "avi", "mov", "mkv", "webm")):
                self.videos.append((os.path.join(self.root_dir, 'UCF-101', video), class_idx))

    def _read_video(self, video_path):
        """
        Use decord to read video frames from the video file.
        Args:
            video_path (str): Path to the video file.
        Returns:
            torch.Tensor: Tensor containing the video frames.
        """
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()

        total_frames = len(vr)
        frame_interval = int(round(fps))
        if self.fps == 1:
            frame_indices = list(range(0, total_frames, frame_interval))
        else:
            frame_indices = list(range(total_frames))
        video = vr.get_batch(frame_indices)
        return video

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Get a video tensor at the given index.
        """
        video_path, target = self.videos[idx]
        video = self._read_video(video_path)
        # get middle frame
        middle_image = video[len(video) // 2]
        if self.transform is not None:
            middle_image = self.transform(middle_image)
        return middle_image, target