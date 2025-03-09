import pandas as pd


def batch_split(inp, batch_size: int):
    if type(inp) is pd.DataFrame:
        inp = inp.to_dict('records')

    cnt = 0
    output = []

    for row in inp:
        if cnt != 0 and cnt % batch_size == 0:
            yield output
            cnt = 0
            output.clear()
        cnt += 1
        output.append(row)
    yield output


def get_batch_pairs(
        data: pd.DataFrame,
        batch_size: int,
        src_lang: str,
        dst_lang: str,
        learn_both_direction: bool = True
):
    def generator():
        idx = 0
        straight = True
        n = data.shape[0]

        while True:
            srcs, dsts = [], []
            for _ in range(batch_size):
                row = data.iloc[idx % n]
                srcs.append(row[src_lang])
                dsts.append(row[dst_lang])
                idx += 1

            # return either straight (src->dst) or backward (dst->src) direction
            if straight:
                yield srcs, dsts, src_lang, dst_lang
            else:
                yield dsts, srcs, dst_lang, src_lang

            # if learn both -> swap straight parameter
            if learn_both_direction:
                straight = not straight
    return generator()
