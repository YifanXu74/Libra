import webdataset as wds
from libra.models.libra.tokenization_libra import LibraTokenizer
import random
from PIL import Image

I2T_INSTRUCTION_LIST = [
    "Describe the image concisely.",
    "Provide a brief description of the given image.",
    "Offer a succinct explanation of the picture presented.",
    "Summarize the visual content of the image.",
    "Give a short and clear explanation of the image.",
    "Share a concise interpretation of the image provided.",
    "Present a compact description of the photo's key features.",
    "Relay a brief, clear account of the picture shown.",
    "Render a clear and concise summary of the photo.",
    "Write a terse but informative summary of the picture.",
    "Create a compact narrative representing the image presented.",
    "",
]


T2I_INSTRUCTION_LIST = [
    "Generate an image corresponding to the caption.",
    "Create a visual representation of the given description.",
    "Craft an image based on the provided text.",
    "Produce an illustrative depiction of the caption.",
    "Generate an image that reflects the essence of the given text.",
    "Create an accompanying image for the provided description.",
    "Craft a visual interpretation of the given caption.",
    "Generate an image that captures the meaning conveyed in the text.",
    "Create a corresponding image for the given textual context.",
    "",
]

MAX_LENGTH = 2048


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


class LaionDataset(wds.DataPipeline):
    def __init__(self, vis_processor, text_processor, location, tokenizer_name=None, i2t_prob=None, num_img_tokens=578, deterministic=True, pad_to_square_i2t=False, add_newline_sep=False, label_mask_strategy="prompt", use_instruction=False, shape_ratio_threshold=None, continuous_prob_t2i=0.0, min_image_size=0, large_init=False, **kwargs):
        if large_init:
            max_bench=20000
            initial=5000
        else:
            max_bench=5000
            initial=1000
        super().__init__(
            wds.ResampledShards(location, deterministic=deterministic),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(max_bench, initial=initial, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map(self.post_process, handler=wds.warn_and_continue),
        )

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.i2t_prob = i2t_prob
        self.num_img_tokens = num_img_tokens
        self.pad_to_square_i2t = pad_to_square_i2t
        self.add_newline_sep = add_newline_sep
        self.label_mask_strategy = label_mask_strategy
        self.use_instruction = use_instruction
        self.shape_ratio_threshold = shape_ratio_threshold
        self.continuous_prob_t2i = continuous_prob_t2i
        self.min_image_size = min_image_size
        
        self.tokenizer = LibraTokenizer.init_text_tokenizer(tokenizer_name)
        self.custom_configs = kwargs
    
    # def cal_num_img_tokens(self, resolution):
    #     vector_resolution = resolution // self.vision_downsample_ratio
    #     if self.tokenizer_version == "v2":
    #         num_img_tokens = vector_resolution**2 + vector_resolution + 1
    #     else:
    #         num_img_tokens = vector_resolution**2 + 2
    #     return num_img_tokens

    def select(self, sample):
        return sample
    
    def shape_check(self, image):
        width, height = image.size
        if max(width, height) < self.min_image_size:
            raise ValueError("Too small image size.")
        
        if width==0 or height==0:
            raise ValueError("Invalid image: zero width/height.")
        if self.shape_ratio_threshold is not None:
            min_threshold = min(self.shape_ratio_threshold, 1/self.shape_ratio_threshold)
            max_threshold = max(self.shape_ratio_threshold, 1/self.shape_ratio_threshold)
            acceptable_shape = (min_threshold < width/height < max_threshold)
        else:
            acceptable_shape = True
        return acceptable_shape


    def post_process(self, sample):
        caption_ = self.text_processor(sample[1]["caption"])

        image = sample[0]
        acceptable_shape = self.shape_check(image)


        label_mask_strategy = self.label_mask_strategy
        use_instruction = self.use_instruction

        if not acceptable_shape:
            # assert self.i2t_prob > 0., "Meet unacceptable shapes but no i2t."
            p = 0. # do not conduct t2i under extreme shape
        else:
            p = random.random()

        if p < self.i2t_prob:
            # i2t
            if self.pad_to_square_i2t or (not acceptable_shape):
                if hasattr(self.vis_processor.transform, "image_mean"):
                    background_color = tuple(int(x*255) for x in self.vis_processor.transform.image_mean)
                else:
                    background_color = (0, 0, 0)
                image = expand2square(image, background_color=background_color)
            image = self.vis_processor(image)
            num_img_tokens = self.num_img_tokens

            background = (" <img_ph>" * num_img_tokens).strip() # add image placeholders to text 
            instuction = random.choice(I2T_INSTRUCTION_LIST) if use_instruction else ""
            respond = caption_
            img_type = "background"

            contiguous_ignore_sign = False

        else:
            # t2i
            image = self.vis_processor(image)
            num_img_tokens = self.num_img_tokens

            background = caption_
            instuction = random.choice(T2I_INSTRUCTION_LIST) if use_instruction else ""
            respond = (" <img_ph>" * num_img_tokens).strip() # add image placeholders to text 
            img_type = "respond"

            p_c = random.random()
            if p_c < self.continuous_prob_t2i:
                contiguous_ignore_sign = False
            else:
                contiguous_ignore_sign = True

            

        caption, label_mask_position_map = self.process_caption(background=background,
                                                                respond=respond,
                                                                instuction=instuction,
                                                                label_mask_strategy=label_mask_strategy,
                                                                img_type=img_type,
                                                                )

        return {
            "vision": image,
            "language": caption,
            "label_mask_position_map": label_mask_position_map,
            "contiguous_ignore_sign": contiguous_ignore_sign,
        }

    def process_caption(self, background: str, respond: str, instuction: str = "", label_mask_strategy = "prompt", img_type="respond"):
        '''
        All should be stripped.
        Only support image-text pairs!
        '''
        assert img_type in ["background", "respond"]

        background = background.strip()
        respond = respond.strip()
        instuction = instuction.strip()

        # if instuction:
        #     instuction_ = instuction + "\n"
        # else:
        #     instuction_ = "\n"

        # instuction_ = instuction + "\n"

        if instuction:
            if img_type == "background":
                instuction_ = "\n" + instuction + "\n"
            elif img_type == "respond":
                instuction_ = " " + instuction + "\n "
            else:
                raise NotImplementedError
        else:
            if img_type == "background":
                instuction_ = "\n"
            elif img_type == "respond":
                instuction_ = "\n "
            else:
                raise NotImplementedError
        
        if self.add_newline_sep and img_type == "background":
            respond = respond + "\n"

        caption = background + instuction_ + respond
        tokenized = self.tokenizer(caption, return_length=True)

        label_mask_position_map = []
        if label_mask_strategy == "prompt":
            start_idx = 0
            end_idx = tokenized.char_to_token(len(background + instuction_))
            label_mask_position_map.append((start_idx, end_idx))
        elif label_mask_strategy == "instruction":
            start_idx = tokenized.char_to_token(len(background))
            end_idx = tokenized.char_to_token(len(background + instuction_))
            label_mask_position_map.append((start_idx, end_idx))
        elif label_mask_strategy == "none":
            pass
        else:
            raise NotImplementedError

        # mask the nearest text token after a image
        if img_type == "respond":
            start_idx = tokenized.length[0] - 1 if self.tokenizer.add_eos_token else tokenized.length[0]
            end_idx = start_idx + 1
            label_mask_position_map.append((start_idx, end_idx))
        elif img_type == "background":
            start_idx = tokenized.char_to_token(len(background))
            end_idx = start_idx + 1
            label_mask_position_map.append((start_idx, end_idx))
        
        return caption, label_mask_position_map
    

    def collater(self, samples):
        keys = samples[0].keys()
        new_samples = {key: [] for key in keys}
        for sample in samples:
            for key, value in sample.items():
                if key == "vision" and value is None:
                    continue
                if key == "contiguous_ignore_sign" and value is None:
                    continue
                new_samples[key].append(value)
        if "return_loss" in new_samples.keys():
            new_samples.pop("return_loss") # for eval
        return {"samples": new_samples, "return_loss": True}