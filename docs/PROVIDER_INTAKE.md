# Questions for a new data provider

Two questions. Everything else (Linux, GPU, disk, network access) is stated as
a requirement in the email we send, and the install checks it and explains
any problem in plain language.

1. **What does your data look like, and which part is the answer the model
   should predict?** Any format is fine: tables, images, slides, expression
   matrices, event logs. The easiest way to show us is a tiny made-up sample in
   the same format (a few rows or files with fake values), or a description of
   the files and roughly how big they are.
2. **Does the data describe people (for example, patients)?**

## What the answers mean for us

| Answer | Action |
|---|---|
| Q1 | Write their task from the closest template in `examples/tasks/` (see `docs/OPERATOR.md`). We adapt to their data, never the reverse. If models for this task may be large, set `limits.gpus` and `gpu_memory_gb`: the email then asks for a GPU machine |
| Q2 yes | Lower `submissions_total`, and ask them to enable gVisor (see the guide's FAQ) |

If a provider tells us their machine can't connect out to the internet, or
they only have a shared university cluster, that isn't supported yet.
