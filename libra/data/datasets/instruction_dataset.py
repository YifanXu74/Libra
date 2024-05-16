import os

from PIL import Image
import torch
from libra.models.libra.tokenization_libra import LibraTokenizer
from libra.data.datasets import conversation as conversation_lib
from torch.utils.data import Dataset
import json
import copy

DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
MODEL_MAX_LENGTH = 2048

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, ann_path: str,
                 vis_processor,
                 version,
                 tokenizer_name,
                 num_img_tokens,
                 vis_root,
                 enable_t2i,
                 image_size,
                 **kwargs):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(ann_path, "r"))
        self.vis_root = vis_root
        self.vis_processor = vis_processor

        self.pad_to_square_i2t = kwargs.get("pad_to_square_i2t", False)

        self.modeling_image = kwargs.get("modeling_image", False)

        self.num_img_tokens = num_img_tokens
        self.image_size = image_size
        self.version = version
        self.enable_t2i = enable_t2i

        if version in conversation_lib.conv_templates:
            self.conversation = conversation_lib.conv_templates[version]
        else:
            raise NotImplementedError

        self.tokenizer = LibraTokenizer.init_text_tokenizer(tokenizer_name)
        self.data_args = kwargs

        if self.version == "plain":
            for data_dict in self.list_data_dict:
                conversation = data_dict["conversations"]
                assert len(conversation) == 2
                assert DEFAULT_IMAGE_TOKEN in conversation[0]['value']
                data_dict["conversations"][0]["value"] = DEFAULT_IMAGE_TOKEN

        if self.enable_t2i:
            assert self.version == "plain"
            list_data_dict_temp = copy.deepcopy(self.list_data_dict)
            new_t2i_data_dict = []
            for data_dict in list_data_dict_temp:
                role_0 = data_dict["conversations"][0]["from"]
                role_1 = data_dict["conversations"][1]["from"]

                data_dict["conversations"] = [data_dict["conversations"][1], data_dict["conversations"][0]]
                
                data_dict["conversations"][0]["from"] = role_0
                data_dict["conversations"][1]["from"] = role_1
                data_dict["task"] = "text2image"
                new_t2i_data_dict.append(data_dict)

            self.list_data_dict = self.list_data_dict + new_t2i_data_dict

    
    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = self.num_img_tokens if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def __getitem__(self, i):
        ########### dddebug ##################
        # i = random.choice([390914, 632530])
        # i = 632530
        # [390914, 632530]
        # print("id: {}".format(i))
        #################################


        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_path = os.path.join(self.vis_root, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                try: 
                    dir_name = os.path.dirname(image_path)
                    file_name = os.path.basename(image_path)
                    file_name = file_name.replace("-", "_")
                    image_path = os.path.join(dir_name, file_name)
                    image = Image.open(image_path).convert('RGB')
                    print("File name incorrect: {}".format(str(image_path)))
                except:
                    print("File not found: {}".format(str(image_path)))
                    return None
            task = self.list_data_dict[i].get("task")
            if self.data_args.get("image_aspect_ratio") == 'pad' and task != "text2image":
                def expand2square(pil_img, background_color=(0,0,0)):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                    
                if hasattr(self.vis_processor.transform, "image_mean"):
                    background_color = tuple(int(x*255) for x in self.vis_processor.transform.image_mean)
                else:
                    background_color = (0, 0, 0)

                image = expand2square(image, background_color=background_color)
                image = self.vis_processor(image)
            else:
                image = self.vis_processor(image)

            if task == "text2image":
                contiguous_ignore_sign = True
            else:
                contiguous_ignore_sign = False

            sources = self.preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                )
            has_image = True
        else:
            image = torch.zeros(3, self.image_size, self.image_size)
            sources = self.preprocess_for_safe(
                copy.deepcopy([e["conversations"] for e in sources]),
                )
            # sources = copy.deepcopy([e["conversations"] for e in sources])
            has_image = False
            contiguous_ignore_sign = None

        processed_text = self.preprocess(sources)
        
        conversation, label_mask_position_map = processed_text


        return {
            "vision": image,
            "language": conversation,
            "label_mask_position_map": label_mask_position_map,
            "has_image": has_image,
            "contiguous_ignore_sign": contiguous_ignore_sign,
        }

    def preprocess(
        self,
        sources,
    ):
        """
        Given a list of sources, each is a conversation list. This transform:
        1. Add signal '### ' at the beginning each sentence, with end signal '\n';
        2. Concatenate conversations together;
        3. Tokenize the concatenated conversation;
        4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
        """
        if self.conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
            return self.preprocess_plain(sources)
        if self.conversation.version.startswith("v1"):
            return self.preprocess_v1(sources)


    def preprocess_for_safe(
        self,
        sources
    ):
        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '')
        return sources

    def preprocess_multimodal(
        self,
        sources
    ):
        has_image = False
        capitalize = self.data_args.get("capitalize", False)
        for source in sources:
            for sentence in source:
                if capitalize:
                    sentence['value'] = sentence['value'].capitalize()
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    if self.version == "v1":
                        assert sentence['from'] == "human", "Images are only supported as instructions during instruction tuning in current version."

                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = (" <img_ph>" * self.num_img_tokens).strip() + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    has_image = True
        assert has_image
        return sources
    
    def preprocess_plain(
        self,
        sources
    ):
        assert len(sources) == 1
        source = sources[0]
        assert len(source) == 2

        img_type = []
        if DEFAULT_IMAGE_TOKEN in source[0]["value"] or "<img_ph>" in source[0]["value"]:
            img_type += ["background"]   
        if DEFAULT_IMAGE_TOKEN in source[1]["value"] or "<img_ph>" in source[1]["value"]:
            img_type += ["respond"]    
        assert len(img_type) == 1
        img_type = img_type[0]

        if img_type == "background":
            background = (" <img_ph>" * self.num_img_tokens).strip()
            instruction = "\n"
            respond = source[1]['value'] + self.conversation.sep
        elif img_type == "respond":
            background = source[0]['value']
            instruction = "\n "
            respond = (" <img_ph>" * self.num_img_tokens).strip() + self.conversation.sep

        caption, label_mask_position_map = self.process_caption(
                                                                background=background,
                                                                respond=respond,
                                                                instruction=instruction,
                                                                label_mask_strategy="prompt",
                                                                img_type=img_type,
                                                            )
        return caption, label_mask_position_map
    

    def preprocess_v1(
        self,
        sources,
    ):
        conv = self.conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            
            conversations.append(conv.get_prompt())
        
        assert len(conversations) == 1
        conversation = conversations[0]
        tokenized = self.tokenizer(conversation, return_length=True)

        # Mask targets
        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
        sep = conv.sep + conv.roles[1] + ": "

        label_mask_position_map = []
        rounds = conversation.split(conv.sep2)
        label_mask_position_map.append((0, 1)) # mask <s>
        cur_len = 0
        for rou in rounds:
            if rou == "":
                break
            rou += conv.sep2
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if self.modeling_image:
                if "<img_ph>" in parts[0]:
                    assert parts[0].startswith(conv.system + " USER: <img_ph> ")
                    start_idx = tokenized.char_to_token(cur_len)
                    end_idx = tokenized.char_to_token(cur_len + len(conv.system + " USER: <img_ph> "))
                    label_mask_position_map.append((start_idx, end_idx))

                    start_idx = tokenized.char_to_token(cur_len + len(conv.system + " USER: " + ("<img_ph> "*self.num_img_tokens).strip()))
                    end_idx = tokenized.char_to_token(cur_len + len(parts[0]))
                    label_mask_position_map.append((start_idx, end_idx))
                else:
                    start_idx = tokenized.char_to_token(cur_len)
                    end_idx = tokenized.char_to_token(cur_len + len(parts[0]))
                    label_mask_position_map.append((start_idx, end_idx))
            else:
                start_idx = tokenized.char_to_token(cur_len)
                end_idx = tokenized.char_to_token(cur_len + len(parts[0]))
                label_mask_position_map.append((start_idx, end_idx))

            cur_len += len(rou)

        return conversation, label_mask_position_map
    
        # if has_image:
        #     input_ids = torch.stack([tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
        # else:
        #     input_ids = self.tokenizer(
        #         conversations,
        #         return_tensors="pt",
        #         padding="longest",
        #         max_length=MODEL_MAX_LENGTH,
        #         truncation=True,
        #     ).input_ids

        # targets = input_ids.clone()

        # assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

        # # Mask targets
        # sep = conv.sep + conv.roles[1] + ": "
        # for conversation, target in zip(conversations, targets):
        #     total_len = int(target.ne(self.tokenizer.pad_token_id).sum())

        #     rounds = conversation.split(conv.sep2)
        #     cur_len = 1
        #     target[:cur_len] = IGNORE_INDEX
        #     for i, rou in enumerate(rounds):
        #         if rou == "":
        #             break

        #         parts = rou.split(sep)
        #         if len(parts) != 2:
        #             break
        #         parts[0] += sep

        #         if has_image:
        #             round_len = len(tokenizer_image_token(rou, self.tokenizer))
        #             instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2
        #         else:
        #             round_len = len(self.tokenizer(rou).input_ids)
        #             instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

        #         target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

        #         cur_len += round_len
        #     target[cur_len:] = IGNORE_INDEX

        #     if cur_len < self.tokenizer.model_max_length:
        #         if cur_len != total_len:
        #             target[:] = IGNORE_INDEX
        #             print(
        #                 f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #                 f" (ignored)"
        #             )

        # return dict(
        #     input_ids=input_ids,
        #     labels=targets,
        # )
    

    def process_caption(self, background: str, respond: str, instruction: str, label_mask_strategy: str, img_type: str):
        '''
        All should be stripped.
        Only support image-text pairs!
        '''
        assert img_type in ["background", "respond"]


        caption = background + instruction + respond
        tokenized = self.tokenizer(caption, return_length=True)

        label_mask_position_map = []
        if label_mask_strategy == "prompt":
            start_idx = 0
            end_idx = tokenized.char_to_token(len(background + instruction))
            label_mask_position_map.append((start_idx, end_idx))
        elif label_mask_strategy == "instruction":
            start_idx = tokenized.char_to_token(len(background))
            end_idx = tokenized.char_to_token(len(background + instruction))
            label_mask_position_map.append((start_idx, end_idx))
        elif label_mask_strategy == "none":
            pass
        else:
            raise NotImplementedError

        # mask the nearest text token after a image
        if img_type == "respond":
            if caption.endswith("<img_ph>\n"):
                start_idx = tokenized.length[0] - 2 if self.tokenizer.add_eos_token else tokenized.length[0] - 1
                end_idx = start_idx + 1
            elif caption.endswith("<img_ph>"):
                start_idx = tokenized.length[0] - 1 if self.tokenizer.add_eos_token else tokenized.length[0]
                end_idx = start_idx + 1
            else:
                raise NotImplementedError
            label_mask_position_map.append((start_idx, end_idx))
        elif img_type == "background":
            assert background.endswith("<img_ph>")
            start_idx = tokenized.char_to_token(len(background))
            end_idx = start_idx + 1
            label_mask_position_map.append((start_idx, end_idx))
        
        return caption, label_mask_position_map
    
    def collater(self, samples):
        filtered_samples = []
        for s in samples:
            if s is not None:
                filtered_samples.append(s)
        samples = filtered_samples

        keys = samples[0].keys()
        new_samples = {key: [] for key in keys}
        for sample in samples:
            for key, value in sample.items():
                if key == "vision" and value is None:
                    continue
                if key == "contiguous_ignore_sign" and value is None:
                    continue
                new_samples[key].append(value)
        return {"samples": new_samples}


    




