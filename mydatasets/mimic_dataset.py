import os
import torch
import json
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate as pytorch_default_collate
import random
from einops import rearrange

from paths import IMAGES_MIMIC_PATH, DICT_CSV_MIMIC_PATH
from mydatasets.mydatasets_utils import ifcc_clean_report, vilmedic_collate

class mimic_Dataset(Dataset):

    def __init__(self, 
                 transform, 
                 tokenizer,
                 processor,
                 partition = "train",
                 text_preprocessing="ifcc_clean_report",
                 multi_image=2):

        self.transform = transform
        self.tokenizer = tokenizer
        self.processor = processor
        self.partition = partition
        self.text_preprocessing = text_preprocessing if text_preprocessing is None else eval(text_preprocessing)
        self.multi_image = multi_image
        self.random_padding = self.partition == "train"

        # Load CSV partition
        self.csv_path = DICT_CSV_MIMIC_PATH[self.partition]
        self.dataset_df = pd.read_csv(self.csv_path)
        
        # Remove empty question or answer from self.dataset_df
        self.remove_empty_text()

        # Set images path
        self.img_root_dir = pathlib.Path(IMAGES_MIMIC_PATH) if IMAGES_MIMIC_PATH is not None else pathlib.Path.cwd()

        self.bos_token = tokenizer("[BOS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.eos_token = tokenizer("[EOS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.pad_token = tokenizer("[PAD]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.sep_token = self.tokenizer("[SEP]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        self.cls_token = self.tokenizer("[CLS]", return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        print("BOS token: ", self.bos_token.item())
        print("EOS token: ", self.eos_token.item())
        print("PAD token: ", self.pad_token.item())
        print("SEP token: ", self.sep_token.item())
        print("CLS token: ", self.cls_token.item())

        # State for resumability
        self._position = 0
        self._epoch = 0
        self._indices = list(range(len(self.dataset_df)))
        if self.partition == "train":
            random.shuffle(self._indices)

    def __len__(self):
        return len(self.dataset_df)
    
    def clean_bad_ids_rg(self):
        print("Initial number of rows: ", self.dataset_df.shape[0])
        self.l_no_ids_rg = []
        with open(self.path_no_ids_rg, 'r') as file:
            for line in file:
                # Process each line as needed
                self.l_no_ids_rg.append(int(line.strip()))
                
        self.dataset_df.drop(self.l_no_ids_rg, inplace=True)
        print("Number of rows after deleting bad IDs: ", self.dataset_df.shape[0])


    def __getitem__(self, idx):
        if self.partition == "train":
            idx = self._indices[idx]

        #idx = 0
        img_list_from_idx = []

        num_images = len(self.dataset_df.iloc[idx].images.split(","))

        # Process all images from patient idx
        for i in range(num_images):
            img_name = self.img_root_dir / self.dataset_df.iloc[idx].images.split(",")[i]
            image = Image.open(img_name).convert('RGB')
            #image.save('rad.png')
            
            # Apply transformation
            if isinstance(self.transform, transforms.Compose):
                # If torchvision transformation
                image = self.transform(image)
            elif isinstance(self.transform, A.core.composition.Compose):
                # If Albumentations transformation
                image = self.transform(image=np.asarray(image))['image']
            else:
                raise ValueError("Unknown transformation type. Supported types: torchvision.transforms.Compose, albumentations.core.composition.Compose")
            
            # Image Processor
            image = np.array(image)
            image = self.processor(image, 
                                   random_padding=self.random_padding, 
                                   return_tensors="pt",
                                   size=384).pixel_values
            image = image.squeeze()

            # FOR PROCESSOR OF TORCHVISION
            # print("image shape: ", image.shape, 'type: ', type(image))
            # image = np.array(image)
            # image = torch.tensor(image, dtype=torch.float32)
            # image = rearrange(image, 'h w c -> c h w')
            # image = self.processor(image)

            #transforms.ToPILImage()(image[0]).save("trad.jpg")
            #print("max: ", torch.max(image))
            #print("min: ", torch.min(image))
            #print("------------------")

            # print(type(image))
                
            img_list_from_idx.append(image)

        # QA
            # Obtén la pregunta y la respuesta por separado
        question = self.dataset_df.iloc[idx].question
        question = self.text_preprocessing(question)
        
        # Aplica el preprocesamiento al texto de salida (answer)
        answer = self.dataset_df.iloc[idx].answer
        raw_answer = self.text_preprocessing(answer)

        question = self.tokenizer(question,
                                    padding=False,
                                    truncation=True,
                                    max_length=64,
                                    return_tensors="pt",
                                    add_special_tokens=False)["input_ids"][0]
        
        # Pad the question to 64 tokens
        question = torch.nn.functional.pad(question, (0, 64 - len(question) - 1), value=self.pad_token.item())

        # Add special tokens
        question = torch.cat([self.cls_token, question, self.bos_token]) 
        
        question_mask = torch.ones_like(question)
        question_mask[question == self.pad_token] = 0

        answer = self.tokenizer(raw_answer,
                                    padding=False,
                                    truncation=True,
                                    max_length=256,
                                    return_tensors="pt",
                                    add_special_tokens=False)["input_ids"][0]
        
        
        # Pad the answer to 64 tokens   
        answer = torch.nn.functional.pad(answer, (0, 256 - len(answer) - 1), value=self.pad_token.item())
        # Add special tokens
        answer = torch.cat([answer, self.eos_token])
        answer_mask = torch.ones_like(answer)
        answer_mask[answer == self.pad_token] = 0

        # cls_ignore = torch.tensor([-100])

        # image_ignore = torch.tensor([-100] * 288)

        # sep_ignore = torch.tensor([-100])

        question_ignore = torch.tensor([-100] * len(question))

        # labels = torch.cat([cls_ignore,
        #                     image_ignore,
        #                     sep_ignore,
        #                     question_ignore,
        #                     answer])

        labels = torch.cat([question_ignore, answer])
        
        # Calculate images_mask
        im_and_immask = vilmedic_collate([img_list_from_idx], self.multi_image)
        images = im_and_immask["images"]
        images_mask = im_and_immask["images_mask"]
              
        return {'idx': idx, 
                'images': images,
                'images_mask': images_mask,
                'question': question,
                'question_mask': question_mask,
                'answer': answer,
                'answer_mask': answer_mask,
                'raw_answer': raw_answer,
                'labels': labels}
    
    def remove_empty_text(self):
        # Remove rows with empty question or answer
        self.dataset_df.dropna(subset=['question', 'answer'], inplace=True)
        print("Len before removing empty text", len(self.dataset_df))

        # Further remove any rows where the answer or question is effectively empty
        self.dataset_df = self.dataset_df[
            (self.dataset_df['question'].str.strip() != '') & 
            (self.dataset_df['answer'].str.strip() != '')
        ]
        print("Len after removing empty text", len(self.dataset_df))

    def get_collate_fn(self):
        def collate_fn(batch):

            images =  pytorch_default_collate([s['images'] for s in batch])
            images_mask = pytorch_default_collate([s['images_mask'] for s in batch])
            idx = pytorch_default_collate([s['idx'] for s in batch])
            question = pytorch_default_collate([s['question'] for s in batch])
            question_mask = pytorch_default_collate([s['question_mask'] for s in batch])
            answer = pytorch_default_collate([s['answer'] for s in batch])
            answer_mask = pytorch_default_collate([s['answer_mask'] for s in batch])
            raw_answer = [s['raw_answer'] for s in batch]
            labels = pytorch_default_collate([s['labels'] for s in batch])

            collated = {
                'idx': idx,
                'images': images,
                'images_mask': images_mask,
                'questions_ids': question,
                'questions_mask': question_mask,
                'answers_ids': answer,
                'answers_mask': answer_mask,
                'answers': raw_answer,
                'labels': labels
            }
            return collated
        return collate_fn
    
    def state_dict(self):
        return {
            'position': self._position,
            'epoch': self._epoch,
            'indices': self._indices
        }

    def load_state_dict(self, state_dict):
        self._position = state_dict['position']
        self._epoch = state_dict['epoch']
        self._indices = state_dict['indices']
        print(f"Dataset resumed at position {self._position}, epoch {self._epoch}")

