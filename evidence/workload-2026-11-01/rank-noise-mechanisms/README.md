# Bounded measurement-mechanism diagnostics

These are library-only development diagnostics after V2 closed. They do not reopen V2, accept a kernel, establish application acceleration or supply untouched confirmation. Contracts and source hashes were recorded locally before each execution. UTC timestamps in the original records are retained; the folder belongs to the November 1 campaign.

The first experiment used a small process-local Objective-C++ observer to attach completion handlers to this Python process's Metal command buffers. It recorded 10,800 owned buffer records across 340 samples in 4.81 seconds, with no zero timestamps, dropped records or pending callbacks. The observer was restored successfully. It did not capture a system trace or access foreign processes. The implementation follows the [Metal completion/timing API](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime) and the [version-pinned MLX command-buffer path](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp).

An injected 5 ms host sleep changed median wall time from 1,069 to 7,543 microseconds while median GPU-buffer union duration changed from 443 to 446 microseconds. Armed/unarmed observer medians were 1,013/990 microseconds; this is descriptive overhead, not an exact causal estimate. The observed device-duration intervals were wider than wall intervals. Device timestamps did **not** solve the noise. GPU buffer duration includes execution stalls and is not pure kernel time or exclusive device ownership.

The second experiment used no observer. It interleaved original short AB/BA controls with prospectively fixed long ABBA/BAAB controls. Each long rank window contains 32 complete original rank128 bursts; each long application window contains two original application16 blocks. All inner/outer samples and randomized orders are retained. Two A and two B windows fold into one paired group, giving forty groups per comparison. They are not eighty independent samples; environmental independence is not established.

| Identical-library control | Original short primary interval | Long balanced primary interval |
| --- | --- | --- |
| Notes rank | [1.0023, 1.4406] | [0.9964, 1.0160] |
| Manuals rank | [0.9917, 1.2811] | [0.9816, 1.0277] |
| Archive rank | [0.9579, 1.0845] | [0.9718, 1.0053] |
| Complete development application | [0.9850, 1.0908] | [0.9157, 1.0526] |

Primary intervals envelope the same paired IID/block4/block8 sensitivities. No outlier was removed and no favorable-load scheduling occurred. All 320 raw groups/orders/subsamples were replayed. The run took 71.70 seconds and peaked at 220,641,532 MLX bytes, within 120 seconds/2 GiB. Rank uncertainty decreased substantially in this observed experiment; application uncertainty remained wide. Longer dependence, clock changes, contention and other causes are not ruled out. No criterion, source or policy in a completed study was changed.

`probe.py`, `buffer_timer.mm` and `counterbalanced_probe.py` are exact historical execution sources. They used repository-relative `.work/rank-device-timing-20261101` and a temporary native helper, so they are not directly runnable from this evidence directory. Preregistration records preserve the original paths and source/binary hashes. Reacquiring the helper does not guarantee the same binary UUID/hash. Keep original results intact when reproducing in a fresh checkout. The historical rank measurement dependency is SHA-256 `4470b506443feea5cf6a4f22b5fea01338e8a199621b02dd0f45f20f114f5b3a`.

`result.json`/`analysis.json` describe the timestamp experiment. `counterbalanced-result.json`/`counterbalanced-analysis.json` describe the separate long-window experiment. Native logs, exact local preregistrations and analysis source remain adjacent. These diagnostics informed a different application-composition investigation; they are not an automatic new candidate study.
