## Summary

<!-- What does this change and why? Link related issues. -->

## Testing

<!-- How did you verify it? Which backend (none / CPU plugin / GPU / Metal)? -->

- [ ] `swift build --build-tests` succeeds
- [ ] Plugin-free tests pass: `swift test --filter 'StableHLOTests|LazyTensorTests'`
- [ ] With a CPU PJRT plugin: `swift test --no-parallel --filter 'MagmaTests|XLARuntimeTests'`
- [ ] New behavior is covered by tests that check values, not only shapes
- [ ] Public API changes are documented (doc comments, README/CHANGELOG as needed)
