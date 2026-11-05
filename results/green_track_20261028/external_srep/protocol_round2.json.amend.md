Amendment 17:35 UTC (pre-registered before running): the transfer check gave val-5 F1 0.54-0.59 for the pre-trained heads
alone (threshold 0.5 best), so external supervision transfers; because the 20-image fine-tune then lands at the baseline
0.64, two more variants are added: (iv) 'preonly' = pre-trained heads without fine-tune as ensemble members (they never
saw the 20 images; independence is the hypothesis); (v) 'joint' = 3000 updates alternating one srep item and one strong
item (no separate fine-tune stage). Judged exactly as before.
