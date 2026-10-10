"""Runs INSIDE the validate sandbox. No submitted code, no answers, no network.

Parses the untrusted predictions parquet (stdin), checks it against the
public output contract and the input keys, and writes canonical predictions
(see contract.to_canonical) to stdout. A compromised parser here gains
nothing: the sandbox holds no answers, and its only outputs are an
enumerated rejection or arrays the bundle could have produced anyway.

    BMB_INPUTS    inputs.parquet (only key columns are read)
    BMB_CONTRACT  public contract JSON (string)

Exit 0: stdout is canonical predictions.
Exit 3: stdout is {"column": ..., "reason": ...} JSON.
"""
from __future__ import annotations

import io
import json
import os
import sys

EXIT_INVALID = 3


def main() -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from bmb_eval.contract import Contract, invalid, to_canonical, validate_predictions
    from bmb_eval.errors import EvalError

    contract = Contract.from_dict(json.loads(os.environ["BMB_CONTRACT"]), public=True)
    input_keys = pd.read_parquet(os.environ["BMB_INPUTS"], columns=contract.key)
    data = sys.stdin.buffer.read()
    key = ",".join(contract.key)
    try:
        try:
            pf = pq.ParquetFile(io.BytesIO(data))
            # Check declared shape from metadata before decoding any pages.
            if pf.metadata.num_rows > len(input_keys):
                raise invalid(key, "unexpected_keys")
            if pf.metadata.num_rows < len(input_keys):
                raise invalid(key, "missing_keys")
            if pf.metadata.num_columns > len(contract.key) + len(contract.outputs):
                raise invalid("*", "unexpected_column")
            preds = pf.read().to_pandas()
        except EvalError:
            raise
        except Exception:
            raise invalid("*", "unreadable")
        ordered = validate_predictions(contract, preds, input_keys)
    except EvalError as e:
        sys.stdout.write(json.dumps(e.detail))
        sys.exit(EXIT_INVALID)
    sys.stdout.buffer.write(to_canonical(contract, ordered))


if __name__ == "__main__":
    main()
