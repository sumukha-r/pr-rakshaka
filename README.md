# pr-rakshaka 🛡️

*Rakshaka (रक्षक) - guardian, protector in Sanskrit*

A PR pre-reviewer that catches missing tests, likely reuse 
opportunities, and a small explicit catalog of datapath smells 
with evidence for each finding, or says nothing.

## Problem
Senior engineers spend hours in PR review catching the same three 
categories of problems: missing tests, reinvented wheels, and 
hot-path performance smells (bottlenecks in code which incur 
time delays which are expensive in a time sensitive environment). 
This blocks shipping and burns expensive 
reviewer time on mechanical checks that could be automated.

## Checks
1. **Missing Tests** - files changed with no test touch
2. **Possible Reuse** - existing symbols above similarity threshold  
3. **Hot-path Smells** - explicit catalog of datapath anti-patterns

## Disclaimer
This is a prototype evaluated on public systems-style PRs plus 
mechanical test-omission cases. Metrics are precision and silence 
rate on that set. The eval corpus was not written by the assistant LLM.

## Status
🚧 Under active development
