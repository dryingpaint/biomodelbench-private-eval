# Questions for a new data provider

Ask these before writing their task. The answers decide what we send them.

1. **Where is your data stored?** A server you run, a university computing
   cluster, a cloud account (AWS, Google Cloud, Azure), or somewhere else?
2. **Is that machine running Linux, always on, and can you install software
   on it?** If not, can your IT group give you a Linux VM? (We have a request
   you can forward.)
3. **Can that machine connect out to the internet over HTTPS?** (Nothing needs
   to connect in.)
4. **Does anyone need to approve installing this, such as IT security? Does
   the data describe people (for example, patients)?**
5. **What should the model predict, from what?** Please send the column names
   of your data file and roughly how many rows it has.

## What the answers mean for us

| Answer | Action |
|---|---|
| Q2 yes, Q3 yes | Standard: write the task, send the email |
| Q2 no | They forward the VM request in the guide to IT first |
| Q1 university cluster, no Linux machine they control | Not supported yet (would need Apptainer); ask research computing for a VM |
| Q3 no | Not supported yet (fully offline sites) |
| Q4 formal review | Send `docs/PROVIDER_SETUP.md` (FAQ section) and `docs/THREAT_MODEL.md` |
| Q4 data about people | Lower `submissions_total`; ask them to enable gVisor |
| Q5 | Use their column names in `bmb-hub new-task`. Regression and yes/no classification are supported |
