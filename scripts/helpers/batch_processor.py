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

