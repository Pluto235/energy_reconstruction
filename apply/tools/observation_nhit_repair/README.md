# Isolated January recovery

This is a frozen fork of `observation_nhit_scalar` for the four terminal failures,
not a replacement of the live/archived production package. Run only `run_repair.py`
through Slurm after the original production and header investigation terminate.
The C++ Nhit reader, transport, quota and checkpoint mechanisms are retained.

`policy.py` pins the authorization SHA and the two exact file/segment identities.
The source and ETO both verify the original input snapshot and authorization. Only
those segments may have empty reduced-header UUID/digest, with explicit missing
status and `poolused_source=user_confirmed_override`. All other segments require
actual headers. The Jan23/Jan28 recovered headers must also match the independent
investigation's real UUID and scalar-header digest. Data exceptions are not changed
into retries or a global poolused default.

Nine fixture groups, including unauthorized segments, changed input snapshots,
forged UUID/digest, missing ordinary headers and full ROOT exception roundtrip,
passed in Slurm 66166. Each source worker still runs the original C++ selfcheck.
Four deterministic shards and exclusive receiver/shard locks prevent duplicate
ownership. A repair receipt is published before source cleanup; only accepted
repair receipts are linked into the primary catalogue. Original failure records
and initial terminal audit remain under the primary state. A separate full-month
Slurm audit must pass before month completion is declared.
