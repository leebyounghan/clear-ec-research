# The `perl alarm` guard does not stop kubectl (lane GPU 2, 2026-11-07)

If your waiters or loop checks use `perl -e 'alarm N; exec @ARGV' kubectl exec ...`, they have no time limit, and a stuck exec stream will hang them.

Measured from this Mac against the pod:

| Command | Limit | Exit code | Took |
|---|---:|---:|---:|
| `alarm 4; exec kubectl exec -- sleep 15` | 4 s | 0 | 18 s |
| Replacement below, same command | 4 s | 124 | 6 s |

The likely cause is that kubectl ignores SIGALRM. With `exec`, the alarm is delivered to kubectl and nothing else is left to act on it. A check of mine hung past 110 s this way at about 02:17 EST.

Replacement: perl stays the parent and sends TERM, then KILL. Normal runs return in the usual time, and exit codes pass through (tested with exit 7).

```bash
perl -e 'my $t=shift;my $p=fork;if(!$p){exec @ARGV or exit 127}$SIG{ALRM}=sub{kill "TERM",$p;sleep 2;kill "KILL",$p;exit 124};alarm $t;waitpid($p,0);exit($?>>8)' 90 kubectl exec POD -- CMD
```

Do not edit a waiter script while it runs: stop it first, or write a new file.

Jobs running on the pod are not affected by this; only local waiters and checks are.
