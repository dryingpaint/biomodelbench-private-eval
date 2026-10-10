# Questions for a new data provider

Ask these before writing their task. The answers decide what we send them.

1. **Where is your data stored?** A server you run, a university computing
   cluster, a cloud account (Google Cloud, AWS, Azure), or somewhere else?
2. **Is that machine running Linux, always on, and can you install software
   on it? Does it have an NVIDIA GPU, and if so, which one?** If it isn't
   suitable, can your IT group give you a Linux VM? (We have a request you
   can forward.)
3. **Can that machine connect out to the internet over HTTPS?** Nothing needs
   to connect in.
4. **Does anyone need to approve installing this, such as IT security? Does
   the data describe people (for example, patients)?**
5. **What does your data look like, and which part is the answer the model
   should predict?** Any format is fine: tables, images, slides, expression
   matrices, event logs. The easiest way to show us is a tiny made-up sample in
   the same format (a few rows or files with fake values), or a description of
   the files and roughly how big they are.

## What the answers mean for us

| Answer | Action |
|---|---|
| Q2 and Q3 yes | Standard: write the task, send the email |
| Q2: no suitable machine | They forward the VM request in the guide to IT first |
| Q2: no GPU, but the task expects large models | A GPU VM (their cloud or IT), or set the task to CPU-only and accept small models |
| Q1: university cluster only | Not supported yet (would need Apptainer); ask research computing for a VM |
| Q3 no | Not supported yet (fully offline sites) |
| Q4: formal review | Send `docs/PROVIDER_SETUP.md` (FAQ section) and `docs/THREAT_MODEL.md` |
| Q4: data about people | Lower `submissions_total`; ask them to enable gVisor |
| Q5 | Write their task from the closest template in `examples/tasks/` (see `docs/OPERATOR.md`). We adapt to their data, never the reverse |
