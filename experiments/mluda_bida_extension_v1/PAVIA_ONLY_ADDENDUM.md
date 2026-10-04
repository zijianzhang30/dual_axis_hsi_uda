# User priority change: Pavia only

User requested Shanghai not run for now, and Pavia C use GPUs 6 and 7.
The nine-task launcher was terminated to prevent any further Shanghai starts.
Shanghai B seed2100 and its DataLoader workers were explicitly terminated;
its existing config/log/history files are preserved as incomplete artifacts,
not final results. The other five Shanghai tasks were never started.
No automatic resume or extra training is authorized by this change.

Keep all three already-running Pavia C processes untouched:
2100 and 2101 on GPU6; 2102 on GPU7. Do not restart them or change their frozen
configuration. Evaluate their fixed epoch100 checkpoints in the independent
user-requested diagnostic, paired with historical B at epoch100. The original
200-epoch runs remain unchanged unless the user explicitly asks to stop them.

The first diagnostic launcher failed during import, before any evaluation or
output directory creation, because an external legacy directory shadowed the
generic module name summarize. Resolve that diagnostic-only import by explicit
file path; this does not modify any training process or algorithm.
