# ✅ Unit Testing Error Criteria Checklist

## 1. Boundary Value Analysis (BVA)
- [ ] Test minimum and maximum values (e.g., 0 and 120)
- [ ] Test extreme values (e.g., `Integer.MAX_VALUE`, `Float.NaN`)
- [ ] Test just-outside boundaries (e.g., -1, 121)
- [ ] Test off-by-one errors
- [ ] Test combinations of boundary values across fields

## 2. Equivalence Partitioning
- [ ] Test one value from each valid input partition
- [ ] Test one value from each invalid input partition
- [ ] Test empty valid partitions
- [ ] Test combinations across multiple input dimensions

## 3. Error Conditions & Exception Handling
- [ ] Test expected exceptions (e.g., `IllegalArgumentException`)
- [ ] Verify exception messages are clear and actionable
- [ ] Simulate unhandled or unexpected exceptions
- [ ] Test nested exception handling or wrapping
- [ ] Check proper logging or fallback behavior on error

## 4. Null, Empty, Zero, and Default Values
- [ ] Test with `null` inputs
- [ ] Test with empty strings, lists, maps, or sets
- [ ] Test with numeric zero values
- [ ] Test default values from uninitialized fields
- [ ] Distinguish between `null` and `0` if relevant

## 5. Data Type and Format Validation
- [ ] Test with incorrect data types
- [ ] Test cross-type comparisons (e.g., `int` vs `long`)
- [ ] Test invalid formats (e.g., badly formatted dates or strings)
- [ ] Test precision/rounding issues for floating-point values
- [ ] Test special characters and encoding issues (e.g., UTF-8)

## 6. State and Lifecycle Testing
- [ ] Access methods before initialization
- [ ] Call methods after teardown or cleanup
- [ ] Test for correct state transitions
- [ ] Ensure reentrant methods behave correctly
- [ ] Confirm proper resource cleanup (e.g., files, sockets)

## 7. Performance & Stress (Scoped to Unit Testing)
- [ ] Validate basic performance for expected inputs
- [ ] Check for memory leaks or object retention
- [ ] Simulate delays or latency-sensitive paths
- [ ] Ensure loops or recursion terminate as expected

## 8. Security-Related Checks
- [ ] Validate input sanitization (e.g., prevent SQL injection)
- [ ] Test access control enforcement
- [ ] Check that exception messages don’t leak sensitive info
- [ ] Attempt known injection vectors
- [ ] Simulate forged or replayed inputs

## 9. Concurrency and Thread Safety
- [ ] Test for race conditions with shared resources
- [ ] Simulate deadlocks with concurrent access
- [ ] Ensure atomic operations where needed
- [ ] Verify isolation between tests

## 10. Dependency and Integration Behavior (Mocked/Stubs)
- [ ] Test behavior when dependencies throw errors
- [ ] Simulate unexpected/malformed responses
- [ ] Handle timeouts or delays from dependencies
- [ ] Validate retry/backoff logic

## 11. Time-Dependent Logic
- [ ] Test across time zones and DST boundaries
- [ ] Validate logic around leap years or month transitions
- [ ] Simulate time travel (e.g., expired tokens)
- [ ] Use time-freezing libraries to control the clock

## 12. Configuration and Environment
- [ ] Simulate missing or corrupted config files
- [ ] Test behavior with missing or malformed environment variables
- [ ] Switch locales and languages to test formatting
- [ ] Validate fallbacks to default config values

## 13. Regression Protection
- [ ] Write tests to cover previously fixed bugs
- [ ] Validate behavior during version upgrades
- [ ] Ensure legacy support logic behaves as expected

