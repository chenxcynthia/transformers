# %%
from neel.imports import *
import gc

torch.set_grad_enabled(False)
# %%
output_dir = Path("/workspace/_scratch/induction_mosaic")
output_dir.mkdir(exist_ok=True)

# with open("/workspace/Easy-Transformer/data/models_by_n_params.json", "r") as f:
#     model_names_by_params = json.load(f)
# model_names = list(map(lambda n:n[0], filter(lambda n: "pythia" not in n[0] and n[1]<1e10, model_names_by_params)))
# print(model_names)
model_names = [
    "NeelNanda/Attn_Only_1L512W_C4_Code",
    "NeelNanda/Attn_Only_2L512W_C4_Code",
    "NeelNanda/Attn_Only_3L512W_C4_Code",
    "NeelNanda/Attn_Only_4L512W_C4_Code",
    "NeelNanda/SoLU_1L512W_C4_Code",
    "NeelNanda/SoLU_2L512W_C4_Code",
    "NeelNanda/SoLU_3L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_C4_Code",
    "NeelNanda/GELU_1L512W_C4_Code",
    "NeelNanda/GELU_2L512W_C4_Code",
    "NeelNanda/GELU_3L512W_C4_Code",
    "NeelNanda/GELU_4L512W_C4_Code",
    "distilgpt2",
    "NeelNanda/SoLU_6L768W_C4_Code",
    "gpt2",
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125M",
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/expanse-gpt2-small-x777",
    "NeelNanda/SoLU_8L_v21_old",
    "NeelNanda/SoLU_8L1024W_C4_Code",
    "NeelNanda/SoLU_10L_v22_old",
    "gpt2-medium",
    "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
    "NeelNanda/SoLU_12L_v23_old",
    "NeelNanda/SoLU_12L1536W_C4_Code",
    "gpt2-large",
    "facebook/opt-1.3b",
    "EleutherAI/gpt-neo-1.3B",
    "gpt2-xl",
    "facebook/opt-2.7b",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-j-6B",
    "facebook/opt-6.7b",
]
print(len(model_names))
# %%
# print(torch.cuda.memory_summary())
# for model_name in tqdm.tqdm(['gpt2', 'solu-1l-old']):
for model_name in tqdm.tqdm(model_names):
    print(model_name)
    model = EasyTransformer.from_pretrained(model_name)
    seq_len = 300
    batch_size = 4
    rand_tokens = torch.randint(100, 20000, (batch_size, seq_len))
    try:
        eos_token = model.tokenizer.bos_token_id
    except:
        eos_tokens = 0
    rep_tokens = torch.cat(
        [
            torch.zeros((batch_size, 1), dtype=torch.int64).fill_(eos_token),
            rand_tokens,
            rand_tokens,
        ],
        dim=1,
    ).cuda()
    induction_score_store = torch.zeros(
        (model.cfg.n_heads, model.cfg.n_layers), dtype=torch.float32, device="cuda"
    )

    def induction_pattern_store(attn, hook, layer):
        induction_score_store[:, layer] = attn.diagonal(
            dim1=-2, dim2=-1, offset=1 - seq_len
        ).mean([0, -1])

    _ = model.run_with_hooks(
        rep_tokens,
        fwd_hooks=(
            [
                (
                    f"blocks.{layer}.attn.hook_attn",
                    partial(induction_pattern_store, layer=layer),
                )
                for layer in range(model.cfg.n_layers)
            ]
        ),
        return_type=None,
    )
    fig = imshow(
        induction_score_store,
        yaxis="Head",
        xaxis="Layer",
        title=f"Induction Score for Heads in {model_name}",
        return_fig=True,
        zmax=1.0,
        zmin=0.0,
        color_continuous_scale="Blues",
        color_continuous_midpoint=None,
    )
    fig.write_image(output_dir / f"{model_name.split('/')[-1]}_induction_score.png")
    fig.write_json(output_dir / f"{model_name.split('/')[-1]}_induction_score.json")
    # fig.show()
    del model, rep_tokens, induction_score_store
    gc.collect()

# %%

# model_name = "gpt2"
# model = EasyTransformer.from_pretrained(model_name)
# seq_len = 100
# batch_size = 2
# rand_tokens = torch.randint(100, 20000, (batch_size, seq_len))
# try:
#     eos_token = model.tokenizer.eos_token_id
# except:
#     eos_tokens = 0
# rep_tokens = torch.cat([torch.zeros((batch_size, 1), dtype=torch.int64) + eos_token, rand_tokens, rand_tokens], dim=1).cuda()
# induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), dtype=torch.float32, device='cuda')

# value_weight_caches = {}
# def value_norm_cache(value, hook, layer):
#     value_weight_caches[layer] = value.norm(dim=-1)

# def induction_pattern_store(attn, hook, layer):
#     value_norms = value_weight_caches[layer]# [batch, key_pos]
#     # attn [batch, head_index, query_pos, key_pos]
#     attn = attn * value_norms.permute(0, 2, 1)[:, :, None, :]
#     induction_score_store[layer] = attn.diagonal(dim1=-2, dim2=-1, offset=1-seq_len).mean([0, -1])

# _ = model.run_with_hooks(rep_tokens, fwd_hooks=(
#     [(f"blocks.{l}.attn.hook_v", partial(value_norm_cache, layer=l)) for l in range(model.cfg.n_layers)]+
#     [(f"blocks.{l}.attn.hook_attn", partial(induction_pattern_store, layer=l)) for l in range(model.cfg.n_layers)]), return_type="none")
# imshow(induction_score_store)


# %%
pio.renderers.default = "png"
imshow(torch.randn(20, 20))
# %%
fig1 = imshow(torch.randn(4, 4), return_fig=True)
fig2 = imshow(torch.randn(5, 5), return_fig=True)
fig1.show()
fig2.show()
from plotly.subplots import make_subplots

sfig = make_subplots(1, 2, x_title="Layer", y_title="Head")
sfig.show()
sfig.add_trace(fig1.data[0], row=1, col=1)
sfig.show()
sfig.add_trace(fig2.data[0], row=1, col=2)
sfig.show()
# %%
def official_name_to_alias(official_name):
    for i in loading.MODEL_ALIASES:
        if official_name in i:
            return loading.MODEL_ALIASES[i][0]
    return official_name


# %%
mosaic_dir = Path("/workspace/_scratch/induction_mosaic")
sample_fig = imshow(
    torch.randn(4, 4),
    yaxis="Head",
    xaxis="Layer",
    return_fig=True,
    zmax=1.0,
    zmin=0.0,
    color_continuous_scale="Blues",
    color_continuous_midpoint=None,
)

unit = 250
row = 7
col = 6
n = row * col
traces = []
titles = []
for path in mosaic_dir.iterdir():
    if path.suffix == ".json":
        fig = pio.read_json(path)
        traces.append(fig["data"][0])
        print(path.name)
        model_name = path.name[: -len("_induction_score.json")]
        print(model_name)
        model_name = official_name_to_alias(model_name)
        print(model_name)
        titles.append(model_name)
        if len(traces) == n:
            break
fig = make_subplots(
    rows=row,
    cols=col,
    x_title="Layer",
    y_title="Head",
    subplot_titles=titles,
    horizontal_spacing=0.13 / col,
    vertical_spacing=0.2 / row,
)
fig.layout.coloraxis = sample_fig.layout.coloraxis
for i, trace in enumerate(traces):
    fig.add_trace(trace, row=i // col + 1, col=i % col + 1)
fig.layout.width = col * unit
fig.layout.height = row * unit
fig.layout.title = "Induction Pattern Score for Heads"
fig.layout.title.font.size = 40
fig.layout.title.xanchor = "center"
fig.layout.title.x = 0.5
for ann in fig.layout.annotations:
    ann.font.size = 12
    if ann.text in ["Head", "Layer"]:
        ann.font.size = 30
fig.show()
# %%
fig.write_html("/workspace/_scratch/mosaic.html", include_plotlyjs="cdn")
print("Written to HTML!")
# %%
# fig.layout.width = 1000
# fig.layout.height = 2000
# fig.show()
# %%
