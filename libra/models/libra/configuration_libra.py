from libra.models.llama.configuration_llama import LlamaConfig

class LibraConfig(LlamaConfig):
    model_type = "libra"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # vision part
        vision_down_ratio=4,
        vision_vocab_size=514,
        vision_codebook_num=2,
        max_vision_token_length=578,
        newline_token_id=13,
        vision_embd_pdrop=0.0,
        vision_resid_pdrop=0.0,
        contiguous_signal_size=2048,
        image_feature_resolution=24,
        vision_prediction_mode="1d",
        use_bridge=True,
        bridge_rank=8,
        concat_signals=True,
        norm_signals=True,
        addition_mode=False,
        use_vision_position_embedding=False,
        unified_head=False,
        use_2d_rope=False,
        # language part
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        **kwargs,
    ):
        self.vision_down_ratio=vision_down_ratio
        self.vision_vocab_size=vision_vocab_size
        self.vision_codebook_num=vision_codebook_num
        self.max_vision_token_length=max_vision_token_length
        self.newline_token_id=newline_token_id
        self.vision_embd_pdrop=vision_embd_pdrop
        self.vision_resid_pdrop=vision_resid_pdrop
        self.contiguous_signal_size=contiguous_signal_size
        self.image_feature_resolution = image_feature_resolution
        self.vision_prediction_mode = vision_prediction_mode
        self.use_bridge = use_bridge
        self.bridge_rank = bridge_rank
        self.concat_signals = concat_signals
        self.norm_signals = norm_signals
        self.addition_mode = addition_mode
        self.use_vision_position_embedding = use_vision_position_embedding
        self.unified_head = unified_head
        self.use_2d_rope = use_2d_rope

        self.resid_pdrop=resid_pdrop
        self.attn_pdrop=attn_pdrop
        self.embd_pdrop=embd_pdrop
        super().__init__(
            **kwargs,
        )
