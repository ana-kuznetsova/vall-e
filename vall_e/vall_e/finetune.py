import torch
from torch import Tensor


from functools import partial
from pathlib import Path

from einops import repeat

from ..emb.qnt import decode_to_file
from ..utils import gather_attribute, load_state_dict_non_strict



def example_usage():
    setup_logging(cfg.log_dir)
    device = "cuda"
    train_dl, subtrain_dl, val_dl = create_train_val_dataloader()


    #resps = torch.load("data/test/test.qnt.pt")[0].to(device)
    num_qnts = 1024

    model = NAR(num_qnts).to(device)

    text_list = [
        torch.tensor([2, 3], device=device),
    ]

    x8 = partial(repeat, pattern="t -> t l", l=8)
    proms_list = [
        x8(torch.tensor([2, 3], device=device)),
    ]

    resps_x1_list = [
        resps[:1].t().to(device),
    ]

    resps_x8_list = [
        resps.t().to(device),
    ]

    codes = model(
        text_list,
        proms_list,
        resps_list=resps_x1_list,
        sampling_temperature=0.2,
    )[0]

    decode_to_file(
        codes,
        Path("data/test/test.nar.init.wav"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for i in range(200):
        optimizer.zero_grad()

        _ = model(text_list, proms_list, resps_list=resps_x8_list)

        losses = gather_attribute(model, "loss")
        loss = sum(losses.values())
        loss.backward()

        optimizer.step()

        if i % 20 == 0:
            stats = {k: v.item() for k, v in losses.items()}
            stats["loss"] = loss.item()
            print(f"iter={i}, {stats}.")

    for i in range(1, 8):
        resps_list = [
            resps[:i].t().to(device),
        ]

        codes = model(
            text_list,
            proms_list,
            resps_list=resps_list,
            sampling_temperature=0.2,
        )[0]

        decode_to_file(
            codes,
            Path(f"data/test/test.nar.1-{i}.wav"),
        )


if __name__ == "__main__":
    example_usage()
